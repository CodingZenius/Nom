import os
import sys
import tempfile

_tmp = tempfile.mkdtemp(prefix="nomad-test-")
os.environ["NOMAD_HOME"] = _tmp
os.environ["NOMAD_MD_DIR"] = _tmp
os.environ.pop("DATABASE_URL", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
