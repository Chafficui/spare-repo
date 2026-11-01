"""Die bestehenden Tests nutzen die Platzhalter-Header (X-Role ...) -> Dev-Modus. Die Tests fuer den Codai-Modus
(tests/test_auth.py) setzen ``auth_mode="codai"`` ausdruecklich."""
import os

os.environ.setdefault("COURTVISION_AUTH_MODE", "dev")
