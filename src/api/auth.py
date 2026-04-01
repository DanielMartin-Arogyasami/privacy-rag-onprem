"""
FIX #3: Bearer token authentication middleware.
When ENABLE_AUTH=true, all endpoints require Authorization: Bearer <API_SECRET_KEY>.
"""

from __future__ import annotations

import logging

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import get_settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


def require_auth(
    creds: HTTPAuthorizationCredentials | None = Security(security),
) -> None:
    """Dependency that enforces Bearer token auth when enabled."""
    settings = get_settings()
    if not settings.enable_auth:
        return

    if not creds:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    if creds.credentials != settings.api_secret_key:
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=401, detail="Invalid API key")
