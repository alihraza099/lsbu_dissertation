import sys
from unittest.mock import MagicMock

# Allow api.py to be imported in environments without the full production stack.
# prometheus_client is a runtime dep — tests verify HTTP behaviour, not metrics internals.
if "prometheus_client" not in sys.modules:
    _prom = MagicMock()
    # generate_latest must return bytes so FastAPI can serialise the Response body
    _prom.generate_latest.return_value = b"# Prometheus metrics\n"
    _prom.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"
    sys.modules["prometheus_client"] = _prom
