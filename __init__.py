"""
G2 ML package initialization.
Ensures cymetric package is available in the Python path.
"""
import sys
import pathlib

# Add parent directory to path to find cymetric
_current_dir = pathlib.Path(__file__).parent.absolute()
_parent_dir = _current_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))