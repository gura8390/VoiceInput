"""Stream an equivalent external-data ONNX copy; never instantiate ONNX Runtime.

Large model preparation is coordinated separately from memory measurements.
Only raw_data storage changes. Nodes, attributes, input/output declarations,
initializer dtype/dimensions and every weight payload retain their exact bytes.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import stat

CHUNK = 1024 * 1024


def varint(value):
    if value < 0:
        raise ValueError('A protobuf varint must be nonnegative')
    output = bytearray()
    while value >= 128:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def length_field(number, value):
    return varint(number * 8 + 2) + varint(len(value)) + value


def read_bytes_varint(data, position):
    value = shift = 0
    while position < len(data) and shift < 70:
        item = data[position]
        position += 1
        value |= (item & 127) << shift
        if item < 128:
            return value, position
        shift += 7
    raise ValueError('Truncated or invalid packed protobuf varint')


def absolute_path(path):
    """Check the lexical path before normalization could hide traversal."""
    path = Path(path)
    if '..' in path.parts:
        raise ValueError('Parent traversal is not allowed')
    # Reject Windows alternate data streams as well as drive-relative paths.
    if os.name == 'nt':
        if path.drive and not path.is_absolute():
            raise ValueError('Drive-relative paths are not allowed')
        if any(':' in part for part in path.parts if part != path.anchor):
            raise ValueError('Alternate data streams are not allowed')
        if any(part.rstrip(' .') != part for part in path.parts if part != path.anchor):
            raise ValueError('Ambiguous Windows path components are not allowed')
        reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{n}' for prefix in ('COM', 'LPT') for n in range(1, 10)}
        for part in path.parts:
            if part != path.anchor and (any(char in part for char in '<>"|?*')
                                        or part.split('.')[0].upper() in reserved):
                raise ValueError('Invalid or reserved Windows path component')
    return Path(os.path.abspath(path))


def assert_no_reparse(path):
    """Inspect unresolved components, including dangling links and junctions."""
    path = absolute_path(path)
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError(f'Reparse/symbolic-link path is not allowed: {component}')
    return path


def checked_source(path):
    path = assert_no_reparse(path)
    if not path.is_file():
        raise ValueError('Source model must be an existing regular file')
    return path


def output_paths(source, output, output_root):
    source = checked_source(source)
    output_root = assert_no_reparse(output_root)
    output = assert_no_reparse(output)
    if output_root.exists() and not output_root.is_dir():
        raise ValueError('Output root is not a directory')
    if output == output_root or not output.is_relative_to(output_root) or output.suffix.lower() != '.onnx':
        raise ValueError('Derived ONNX must be below the explicit output root')
    # The input directory and output tree must be disjoint. Never write next
    # to originals, even when the chosen candidate filename is different.
    if source.parent.is_relative_to(output_root) or output_root.is_relative_to(source.parent):
        raise ValueError('Source directory and output root must not overlap')
    paths = (output, output.with_suffix('.weights.bin'), output.with_suffix('.external.json'))
    for path in paths:
        assert_no_reparse(path)
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        raise ValueError('Refusing to overwrite an existing model, weights or receipt')
    return source, output, output_root, paths[1], paths[2]


def guarded_new_file(path, output_root, *, text=False):
    assert_no_reparse(output_root)
    assert_no_reparse(path)
    if not path.is_relative_to(output_root):
        raise ValueError('Output escaped its declared root')
    return path.open('x', encoding='utf-8', newline='\n') if text else path.open('xb')


@dataclass(frozen=True)
class Field:
    number: int
    wire: int
    start: int
    payload: int
    end: int
    scalar: int | None = None


def read_varint(stream):
    result = shift = 0
    while shift < 70:
        raw = stream.read(1)
        if not raw:
            raise ValueError('Truncated protobuf varint')
        result |= (raw[0] & 127) << shift
        if raw[0] < 128:
            return result
        shift += 7
    raise ValueError('Invalid protobuf varint')


def fields(stream, start, end):
    position = start
    while position < end:
        stream.seek(position)
        tag = read_varint(stream)
        number, wire = tag >> 3, tag & 7
        scalar = None
        if not number:
            raise ValueError('Protobuf field zero')
        if wire == 0:
            scalar = read_varint(stream)
            payload, finish = stream.tell(), stream.tell()
        elif wire == 2:
            size = read_varint(stream)
            payload, finish = stream.tell(), stream.tell() + size
        elif wire in (1, 5):
            payload, finish = stream.tell(), stream.tell() + (8 if wire == 1 else 4)
        else:
            raise ValueError('Unsupported protobuf wire type')
        if finish > end:
            raise ValueError('Protobuf field exceeds its message')
        yield Field(number, wire, position, payload, finish, scalar)
        position = finish


def chunks(stream, start, end):
    stream.seek(start)
    while start < end:
        data = stream.read(min(CHUNK, end - start))
        if not data:
            raise ValueError('Truncated file')
        yield data
        start += len(data)


def small_bytes(stream, field):
    if field.end - field.payload > 65536:
        raise ValueError('Unexpectedly large metadata field')
    return b''.join(chunks(stream, field.payload, field.end))


def update_span(digest, stream, start, end):
    for data in chunks(stream, start, end):
        digest.update(data)


def hash_span(stream, start, end):
    digest = hashlib.sha256()
    update_span(digest, stream, start, end)
    return digest.hexdigest()


def sha(path):
    path = checked_source(path)
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect(path, include_payload_hashes=False):
    """Default scanning seeks over raw weights instead of reading them."""
    path = checked_source(path)
    with path.open('rb') as stream:
        top = list(fields(stream, 0, path.stat().st_size))
        graphs = [field for field in top if field.number == 7]
        if len(graphs) != 1 or graphs[0].wire != 2:
            raise ValueError('Expected one ModelProto.graph')
        graph = graphs[0]
        graph_fields = list(fields(stream, graph.payload, graph.end))
        initializer_meta, other_graph, other_model = (hashlib.sha256() for _ in range(3))
        tensors, seen = [], set()
        for field in top:
            if field.number != 7:
                update_span(other_model, stream, field.start, field.end)
        for field in graph_fields:
            if field.number != 5:
                update_span(other_graph, stream, field.start, field.end)
                continue
            inner = list(fields(stream, field.payload, field.end))
            names = [item for item in inner if item.number == 8 and item.wire == 2]
            if len(names) != 1:
                raise ValueError('Initializer name is absent or duplicated')
            name = small_bytes(stream, names[0]).decode('utf-8')
            if name in seen:
                raise ValueError('Duplicate initializer name')
            seen.add(name)
            dtype = [item.scalar for item in inner if item.number == 2 and item.wire == 0]
            if len(dtype) != 1:
                raise ValueError('Initializer dtype missing')
            raw = [item for item in inner if item.number == 9]
            if len(raw) > 1 or any(item.wire != 2 for item in raw):
                raise ValueError('Invalid raw_data field')
            tensor_meta = hashlib.sha256()
            dimensions = []
            external, location = {}, 0
            for item in inner:
                if item.number == 1:
                    if item.wire == 0:
                        dimensions.append(item.scalar)
                    elif item.wire == 2:
                        packed = small_bytes(stream, item)
                        position = 0
                        while position < len(packed):
                            dimension, position = read_bytes_varint(packed, position)
                            dimensions.append(dimension)
                    else:
                        raise ValueError('Invalid tensor dimensions')
                if item.number == 14:
                    if item.wire != 0:
                        raise ValueError('Invalid data_location')
                    location = item.scalar
                elif item.number == 13:
                    entries = list(fields(stream, item.payload, item.end))
                    entry = {part.number: small_bytes(stream, part).decode('utf-8') for part in entries}
                    if set(entry) != {1, 2} or entry[1] in external:
                        raise ValueError('Invalid external_data entry')
                    external[entry[1]] = entry[2]
                elif item.number != 9:
                    update_span(tensor_meta, stream, item.start, item.end)
            encoded_name = name.encode('utf-8')
            initializer_meta.update(varint(len(encoded_name)) + encoded_name + tensor_meta.digest())
            record = {'name': name, 'dtype': dtype[0], 'dimensions': dimensions,
                      'metadata_sha256': tensor_meta.hexdigest(),
                      'raw_bytes': 0 if not raw else raw[0].end - raw[0].payload,
                      'data_location': location, 'external': external,
                      'field': field, 'inner': inner, 'raw_field': raw[0] if raw else None}
            if include_payload_hashes and raw:
                record['payload_sha256'] = hash_span(stream, raw[0].payload, raw[0].end)
            tensors.append(record)
        return {'bytes': path.stat().st_size, 'graph': graph, 'top': top, 'graph_fields': graph_fields,
                'tensors': tensors, 'initializer_metadata_sha256': initializer_meta.hexdigest(),
                'other_graph_fields_sha256': other_graph.hexdigest(),
                'other_model_fields_sha256': other_model.hexdigest()}


def summary(view):
    return {key: view[key] for key in ('bytes', 'initializer_metadata_sha256',
            'other_graph_fields_sha256', 'other_model_fields_sha256')} | {
        'initializer_count': len(view['tensors']),
        'raw_bytes_by_dtype': dict(counter_sum((item['dtype'], item['raw_bytes']) for item in view['tensors'])),
        'external_initializer_count': sum(bool(item['external']) for item in view['tensors']),
        'external_files': sorted({item['external']['location'] for item in view['tensors'] if item['external']}),
    }


def counter_sum(rows):
    result = Counter()
    for key, value in rows:
        result[key] += value
    return result


def external_fields(filename, offset, length):
    entries = {'location': filename, 'offset': str(offset), 'length': str(length)}
    return b''.join(length_field(13, length_field(1, key.encode()) + length_field(2, value.encode()))
                    for key, value in entries.items()) + varint(14 * 8) + varint(1)


def prepare(source, output, *, output_root, minimum=4096, alignment=4096, expected_sha=None):
    source, output, output_root, data_path, receipt_path = output_paths(source, output, output_root)
    if minimum < 1 or alignment < 1 or alignment > 65536 or alignment & (alignment - 1):
        raise ValueError('Invalid threshold/alignment')
    if expected_sha is not None:
        expected_sha = expected_sha.lower()
        if len(expected_sha) != 64 or set(expected_sha) - set('0123456789abcdef'):
            raise ValueError('Expected source SHA256 must contain 64 hexadecimal characters')
    before_stat = source.stat()
    original = inspect(source)
    source_sha = sha(source)
    if expected_sha is not None and source_sha != expected_sha:
        raise ValueError('Source model hash does not match the accepted model')
    planned, total = {}, 0
    for tensor in original['tensors']:
        if tensor['external'] or tensor['data_location']:
            raise ValueError('Source must use inline initializers only')
        if tensor['raw_bytes'] < minimum:
            continue
        total = (total + alignment - 1) & ~(alignment - 1)
        extra = external_fields(data_path.name, total, tensor['raw_bytes'])
        remaining = [item for item in tensor['inner'] if item.number not in (9, 14)]
        new_length = sum(item.end - item.start for item in remaining) + len(extra)
        planned[tensor['field'].start] = (tensor, total, remaining, extra, new_length)
        total += tensor['raw_bytes']
    if not planned:
        raise ValueError('No eligible raw_data initializer')
    assert_no_reparse(output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert_no_reparse(output.parent)
    identities = []
    with source.open('rb') as src, guarded_new_file(data_path, output_root) as data:
        for tensor, offset, _, _, _ in planned.values():
            data.write(bytes(offset - data.tell()))
            digest = hashlib.sha256()
            raw = tensor['raw_field']
            for piece in chunks(src, raw.payload, raw.end):
                data.write(piece)
                digest.update(piece)
            identities.append({'name': tensor['name'], 'dtype': tensor['dtype'],
                               'dimensions': tensor['dimensions'], 'offset': offset, 'bytes': tensor['raw_bytes'],
                               'payload_sha256': digest.hexdigest()})
    graph_length = sum((len(varint(5 * 8 + 2)) + len(varint(planned[field.start][4]))
                        + planned[field.start][4]) if field.start in planned else field.end - field.start
                       for field in original['graph_fields'])
    with source.open('rb') as src, guarded_new_file(output, output_root) as dst:
        for field in original['top']:
            if field.number != 7:
                for piece in chunks(src, field.start, field.end):
                    dst.write(piece)
                continue
            dst.write(varint(7 * 8 + 2) + varint(graph_length))
            for item in original['graph_fields']:
                if item.start not in planned:
                    for piece in chunks(src, item.start, item.end):
                        dst.write(piece)
                    continue
                _, _, remaining, extra, new_length = planned[item.start]
                dst.write(varint(5 * 8 + 2) + varint(new_length))
                for inner in remaining:
                    for piece in chunks(src, inner.start, inner.end):
                        dst.write(piece)
                dst.write(extra)
    derived = inspect(output)
    for key in ('initializer_metadata_sha256', 'other_graph_fields_sha256', 'other_model_fields_sha256'):
        if original[key] != derived[key]:
            raise ValueError(f'Semantic payload metadata changed: {key}')
    derived_by_name = {tensor['name']: tensor for tensor in derived['tensors']}
    if set(derived_by_name) != {tensor['name'] for tensor in original['tensors']}:
        raise ValueError('Initializer names changed')
    copied = {tensor['name']: tensor for tensor in identities}
    verified = []
    with source.open('rb') as src, output.open('rb') as dst, data_path.open('rb') as data:
        for tensor in original['tensors']:
            candidate = derived_by_name[tensor['name']]
            if tensor['dtype'] != candidate['dtype'] or tensor['dimensions'] != candidate['dimensions']:
                raise ValueError('Initializer dtype or dimensions changed')
            tensor_source_sha = tensor_candidate_sha = None
            if tensor['raw_field'] is not None:
                if tensor['name'] in copied:
                    block = copied[tensor['name']]
                    tensor_source_sha = block['payload_sha256']
                    expected_external = {'location': data_path.name, 'offset': str(block['offset']),
                                         'length': str(block['bytes'])}
                    if candidate['external'] != expected_external or candidate['data_location'] != 1:
                        raise ValueError('External initializer points to an unexpected block')
                    tensor_candidate_sha = hash_span(data, block['offset'], block['offset'] + block['bytes'])
                else:
                    tensor_source_sha = hash_span(src, tensor['raw_field'].payload, tensor['raw_field'].end)
                    if candidate['raw_field'] is None:
                        raise ValueError('Inline initializer lost its raw_data')
                    tensor_candidate_sha = hash_span(dst, candidate['raw_field'].payload, candidate['raw_field'].end)
                if tensor_source_sha != tensor_candidate_sha:
                    raise ValueError('Written tensor payload differs from source')
            verified.append({'name': tensor['name'], 'dtype': tensor['dtype'],
                             'dimensions': tensor['dimensions'], 'payload_bytes': tensor['raw_bytes'],
                             'source_payload_sha256': tensor_source_sha, 'candidate_payload_sha256': tensor_candidate_sha,
                             'dtype_dimensions_payload_equal': True,
                             'representation': 'external' if tensor['name'] in copied else 'unchanged_inline'})
    assert_no_reparse(source)
    after_stat = source.stat()
    if (before_stat.st_dev, before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns) != (
            after_stat.st_dev, after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns) or sha(source) != source_sha:
        raise ValueError('Source changed during preparation')
    receipt = {'schema': 1, 'prepared': True, 'sessions_created': 0, 'inference_calls': 0,
               'source': str(source), 'candidate': str(output), 'source_sha256': source_sha,
               'output_root': str(output_root),
               'candidate_sha256': sha(output), 'external_weights_sha256': sha(data_path),
               'source_summary': summary(original), 'candidate_summary': summary(derived),
               'payloads_byte_identical': True, 'metadata_byte_identical': True,
               'externalized': identities, 'tensor_verification': verified,
               'verified_tensor_count': len(verified),
               'external_bytes': total, 'alignment': alignment,
               'execution_state': 'prepared_only_no_runtime_session'}
    with guarded_new_file(receipt_path, output_root, text=True) as receipt_file:
        receipt_file.write(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', type=Path, help='Omit for metadata-only scan; provide to prepare a derived model')
    parser.add_argument('--output-root', type=Path, help='Explicit isolated destination tree; required when preparing')
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--minimum', type=int, default=4096)
    parser.add_argument('--alignment', type=int, default=4096)
    args = parser.parse_args()
    if args.output:
        if args.output_root is None:
            parser.error('--output-root is required with --output')
        result = prepare(args.source, args.output, output_root=args.output_root,
                         minimum=args.minimum, alignment=args.alignment,
                         expected_sha=args.expected_source_sha256)
        print(json.dumps({key: result[key] for key in ('prepared', 'sessions_created', 'external_bytes', 'candidate_sha256')}))
    else:
        print(json.dumps(summary(inspect(args.source)), ensure_ascii=True))


if __name__ == '__main__':
    main()
