"""Compare isolated raw encoder outputs; never treats a numerical delta as CER."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[2] / 'evidence/ggml-vulkan-encoder-20261117'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', required=True, type=Path)
    parser.add_argument('--right', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(STUDY.resolve()) and not output.exists(),
            'Output must be new and under this isolated study')
    left, right = read(args.left), read(args.right)
    for report in (left, right):
        require(report.get('execution_passed') and report.get('cleanup_confirmed'),
                'Only completed, cleaned-up diagnostic runs may be compared')
        require(report.get('diagnostic'), 'Reports must be explicitly diagnostic')
    require(left['manifest_sha256'] == right['manifest_sha256']
            and left['workload_sha256'] == right['workload_sha256'], 'Input identity mismatch')
    rows_left, rows_right = left['encoder_embeddings'], right['encoder_embeddings']
    require(len(rows_left) == len(rows_right) == len(left['samples']) == len(right['samples']),
            'Embedding record/sample count mismatch')
    rows = []
    for index, (a, b) in enumerate(zip(rows_left, rows_right)):
        shape = (a['rows'], a['dimension'])
        require(shape == (b['rows'], b['dimension']) and min(shape) > 0, 'Embedding shape mismatch')
        require(a['pcm_sha256'] == b['pcm_sha256'] == left['samples'][index]['pcm_sha256']
                == right['samples'][index]['pcm_sha256'], 'Embedding PCM/order mismatch')
        path_a, path_b = Path(a['path']).resolve(strict=True), Path(b['path']).resolve(strict=True)
        require(all(path.is_relative_to(STUDY.resolve()) for path in (path_a, path_b)),
                'Embedding file outside isolated study')
        require(path_a.stat().st_size == path_b.stat().st_size == shape[0] * shape[1] * 4,
                'Embedding byte size mismatch')
        x = np.fromfile(path_a, dtype='<f4').astype(np.float64)
        y = np.fromfile(path_b, dtype='<f4').astype(np.float64)
        require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite embedding value')
        delta = x - y
        norm_a, norm_b = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        cosine = float(np.dot(x, y) / (norm_a * norm_b)) if norm_a and norm_b else float(norm_a == norm_b)
        rows.append({'index': index, 'phase': left['samples'][index]['phase'],
                     'pcm_sha256': a['pcm_sha256'], 'shape': shape, 'values': int(x.size),
                     'left_sha256': sha(path_a), 'right_sha256': sha(path_b),
                     'exact': bool(np.array_equal(x, y)),
                     'different_values': int(np.count_nonzero(delta)),
                     'max_absolute_error': float(np.max(np.abs(delta))),
                     'mean_absolute_error': float(np.mean(np.abs(delta))),
                     'rmse': float(np.sqrt(np.mean(delta * delta))),
                     'relative_l2_error': float(np.linalg.norm(delta) / norm_a) if norm_a else None,
                     'cosine_similarity': cosine})
    result = {'complete': True, 'left_report_sha256': sha(args.left),
              'right_report_sha256': sha(args.right), 'samples': rows,
              'all_exact': all(row['exact'] for row in rows),
              'scope': 'Encoder numerical comparison only; downstream text and timestamp checks are separate. '
                       'Embedding dumps are diagnostic and excluded from resource/timing medians.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'complete': True, 'samples': len(rows), 'all_exact': result['all_exact'],
                      'worst_cosine_similarity': min(row['cosine_similarity'] for row in rows)}, allow_nan=False))


if __name__ == '__main__':
    main()
