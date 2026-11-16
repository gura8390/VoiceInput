"""Current project/runtime paths shared by development tools.

This file describes the current layout only. Historical receipts keep the
paths recorded at measurement time and are never silently migrated.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = Path('config/settings.json')
HOTWORDS = Path('config/hot.txt')
HOTWORD_RULES = Path('config/hot-rule.txt')
LLAMA = Path('runtime/llama')
ONNX = Path('runtime/onnx')
AUDIO = Path('runtime/audio')
MEDIA = Path('runtime/media')
