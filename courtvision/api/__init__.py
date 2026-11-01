"""CourtVision-API: REST + WebSocket ueber dem Event-Store (siehe :mod:`courtvision.api.app`)."""
from .app import create_app

__all__ = ["create_app"]
