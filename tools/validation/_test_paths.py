"""Import the actual tool groups when using unittest discovery or direct tests."""
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1]
for group in (TOOLS, TOOLS / 'build', TOOLS / 'research', TOOLS / 'models'):
    sys.path.insert(0, str(group))
from project_layout import ROOT
