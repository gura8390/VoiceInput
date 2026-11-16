"""Synthetic protobuf and filesystem boundary tests; no ORT/audio/model load."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from _test_paths import ROOT
import prepare_external_weights as tool


def tensor(name, dtype, payload=None, dimensions=(), typed=None):
    value = tool.length_field(8, name.encode()) + tool.varint(16) + tool.varint(dtype)
    value += tool.length_field(1, b''.join(tool.varint(dimension) for dimension in dimensions))
    if payload is not None:
        value += tool.length_field(9, payload)
    if typed is not None:
        value += tool.length_field(7, b''.join(tool.varint(item) for item in typed))
    return value + tool.length_field(12, b'unchanged documentation')


def model(tensors):
    graph = tool.length_field(1, tool.length_field(4, b'Identity'))
    graph += b''.join(tool.length_field(5, value) for value in tensors)
    graph += tool.length_field(2, b'unchanged graph name')
    return tool.varint(8) + tool.varint(10) + tool.length_field(7, graph) + tool.length_field(2, b'producer')


class ExternalWeightsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='yyur-model-converter-')
        self.root = Path(self.temporary.name)
        self.source = self.root / 'inputs/source.onnx'
        self.source.parent.mkdir()
        self.output_root = self.root / 'outputs'
        self.output = self.output_root / 'model.onnx'
        self.fp16 = bytes(range(128))
        self.quantized = bytes(reversed(range(65)))
        self.source.write_bytes(model([
            tensor('fp16', 10, self.fp16, (64,)),
            tensor('uint8-packed-weight', 2, self.quantized, (65,)),
            tensor('control', 7, b'12345678'),
            tensor('typed-constant', 7, dimensions=(2,), typed=(3, 5)),
        ]))
        self.junctions = []

    def tearDown(self):
        for path in self.junctions:
            if os.path.lexists(path):
                os.rmdir(path)
        self.temporary.cleanup()

    def prepare(self, **kwargs):
        return tool.prepare(self.source, self.output, output_root=self.output_root,
                            minimum=32, alignment=256, **kwargs)

    def junction(self, link, target):
        if os.name != 'nt':
            self.skipTest('Windows directory-junction case')
        result = subprocess.run(['cmd.exe', '/d', '/c', 'mklink', '/J', str(link), str(target)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                creationflags=0x08000000, timeout=10)
        if result.returncode:
            self.skipTest('Native directory junction could not be created')
        self.junctions.append(link)

    def test_all_tensor_bytes_types_shapes_and_nonweight_fields_remain_equal(self):
        original = self.source.read_bytes()
        receipt = self.prepare(expected_sha=hashlib.sha256(original).hexdigest())
        self.assertEqual(receipt['verified_tensor_count'], 4)
        self.assertEqual(receipt['sessions_created'], 0)
        self.assertEqual(receipt['inference_calls'], 0)
        self.assertTrue(all(row['dtype_dimensions_payload_equal'] for row in receipt['tensor_verification']))
        self.assertEqual(receipt['external_bytes'], 321)
        self.assertEqual(self.output.with_suffix('.weights.bin').read_bytes(), self.fp16 + bytes(128) + self.quantized)
        self.assertEqual(self.source.read_bytes(), original)
        view = tool.inspect(self.output)
        self.assertEqual(view['tensors'][0]['external'], {'location': 'model.weights.bin', 'offset': '0', 'length': '128'})
        self.assertEqual(view['tensors'][1]['external']['offset'], '256')
        self.assertEqual(view['tensors'][1]['dtype'], 2)
        self.assertEqual(view['tensors'][1]['dimensions'], [65])
        self.assertEqual(view['tensors'][2]['raw_bytes'], 8)
        self.assertFalse(view['tensors'][3]['external'])
        for key in ('initializer_metadata_sha256', 'other_graph_fields_sha256', 'other_model_fields_sha256'):
            self.assertEqual(receipt['source_summary'][key], receipt['candidate_summary'][key])
        written = json.loads(self.output.with_suffix('.external.json').read_text('utf-8'))
        self.assertEqual(written, json.loads(json.dumps(receipt)))

    def test_missing_source_creates_no_output_directory(self):
        self.source.unlink()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output_root.exists())

    def test_wrong_source_hash_creates_no_outputs(self):
        with self.assertRaises(ValueError):
            self.prepare(expected_sha='0' * 64)
        self.assertFalse(self.output_root.exists())

    def test_powershell_uppercase_hash_is_accepted(self):
        accepted = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.assertEqual(self.prepare(expected_sha=accepted.upper())['source_sha256'], accepted)

    def test_all_three_existing_output_types_are_preserved(self):
        self.output_root.mkdir()
        for suffix in ('.onnx', '.weights.bin', '.external.json'):
            path = self.output.with_suffix(suffix)
            path.write_bytes(b'preserve existing bytes')
            with self.assertRaises(ValueError):
                self.prepare()
            self.assertEqual(path.read_bytes(), b'preserve existing bytes')
            path.unlink()

    def test_repeat_conversion_never_overwrites_successful_artifacts(self):
        self.prepare()
        files = {path.name: path.read_bytes() for path in self.output_root.iterdir()}
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(files, {path.name: path.read_bytes() for path in self.output_root.iterdir()})

    def test_output_outside_explicit_root_is_rejected(self):
        with self.assertRaises(ValueError):
            tool.prepare(self.source, self.root / 'elsewhere/model.onnx', output_root=self.output_root)
        self.assertFalse((self.root / 'elsewhere').exists())

    def test_parent_traversal_is_rejected_before_normalizing(self):
        for keyword, path in (('source', self.source.parent / '../inputs/source.onnx'),
                              ('output', self.output_root / '../outputs/model.onnx'),
                              ('output_root', self.root / 'temporary/../outputs')):
            arguments = dict(source=self.source, output=self.output, output_root=self.output_root)
            arguments[keyword] = path
            with self.assertRaises(ValueError, msg=keyword):
                tool.prepare(**arguments)
        self.assertFalse(self.output_root.exists())

    def test_source_directory_and_output_tree_cannot_overlap(self):
        for root in (self.root, self.source.parent, self.source.parent / 'derived'):
            with self.assertRaises(ValueError):
                tool.prepare(self.source, root / 'candidate.onnx', output_root=root)
        with self.assertRaises(ValueError):
            tool.prepare(self.source, self.source, output_root=self.source.parent)

    def test_output_root_must_be_directory_and_output_must_be_onnx(self):
        self.output_root.write_bytes(b'file')
        with self.assertRaises(ValueError):
            self.prepare()
        self.output_root.unlink()
        with self.assertRaises(ValueError):
            tool.prepare(self.source, self.output.with_suffix('.bin'), output_root=self.output_root)

    def test_external_source_is_not_reprocessed(self):
        self.prepare()
        with self.assertRaises(ValueError):
            tool.prepare(self.output, self.root / 'other/model.onnx', output_root=self.root / 'other')
        self.assertFalse((self.root / 'other').exists())

    def test_duplicate_initializers_and_truncated_protobuf_are_rejected(self):
        same = tensor('same', 10, self.fp16, (64,))
        for data in (model([same, same]), b'\x3a\x7ftruncated', model([tool.length_field(8, b'missing-dtype')])):
            self.source.write_bytes(data)
            with self.assertRaises(ValueError):
                self.prepare()
            self.assertFalse(self.output_root.exists())

    def test_invalid_threshold_alignment_and_no_large_raw_data(self):
        for minimum, alignment in ((0, 4096), (1, 0), (1, 3), (1, 131072), (999999, 4096)):
            with self.assertRaises(ValueError):
                tool.prepare(self.source, self.output, output_root=self.output_root,
                             minimum=minimum, alignment=alignment)
            self.assertFalse(self.output_root.exists())

    def test_source_file_hardlink_collision_is_rejected_without_mutation(self):
        self.output_root.mkdir()
        os.link(self.source, self.output)
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.source.read_bytes(), original)

    def test_output_directory_junction_and_source_directory_junction_are_rejected(self):
        target = self.root / 'target'
        target.mkdir()
        self.junction(self.output_root, target)
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(list(target.iterdir()), [])
        source_alias = self.root / 'source-alias'
        self.junction(source_alias, self.source.parent)
        with self.assertRaises(ValueError):
            tool.inspect(source_alias / self.source.name)

    def test_nested_output_junction_is_rejected(self):
        self.output_root.mkdir()
        target = self.root / 'target'
        target.mkdir()
        alias = self.output_root / 'nested'
        self.junction(alias, target)
        with self.assertRaises(ValueError):
            tool.prepare(self.source, alias / 'model.onnx', output_root=self.output_root)
        self.assertEqual(list(target.iterdir()), [])

    def test_dangling_file_symlink_is_not_treated_as_free_output(self):
        self.output_root.mkdir()
        try:
            self.output.symlink_to(self.root / 'absent.onnx')
        except OSError:
            self.skipTest('File symbolic-link creation unavailable')
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertTrue(self.output.is_symlink())

    def test_concurrent_converters_have_one_exclusive_writer(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = [pool.submit(self.prepare) for _ in range(2)]
        passed, failed = 0, 0
        for result in results:
            try:
                result.result()
                passed += 1
            except (ValueError, FileExistsError):
                failed += 1
        self.assertEqual((passed, failed), (1, 1))
        self.assertTrue(json.loads(self.output.with_suffix('.external.json').read_text('utf-8'))['prepared'])

    def test_cli_requires_explicit_output_root(self):
        result = subprocess.run([sys.executable, str(ROOT / 'tools/models/prepare_external_weights.py'),
                                 '--source', str(self.source), '--output', str(self.output)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'--output-root', result.stderr)
        self.assertFalse(self.output_root.exists())


if __name__ == '__main__':
    unittest.main()
