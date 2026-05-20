"""LDAP / Active Directory authentication."""

from __future__ import annotations

import logging
import re

from forecast_engine.config import settings

logger = logging.getLogger(__name__)

# Characters that must be escaped in LDAP search filters (RFC 4515)
_LDAP_ESCAPE_RE = re.compile(r'([\\*\(\)\x00/])')


def _ldap_escape(value: str) -> str:
    """Escape special characters for safe use in LDAP search filters."""
    return _LDAP_ESCAPE_RE.sub(lambda m: '\\' + format(ord(m.group(1)), '02x'), value)


async def ldap_authenticate(username: str, password: str) -> dict | None:
    """Bind to AD and return user info dict, or None on failure.

    Returns: {"username": ..., "display_name": ..., "email": ..., "groups": [...]}
    """
    if not settings.ldap_server:
        return None

    # Input validation
    if not username or not password:
        return None

    try:
        import ldap3
        server = ldap3.Server(settings.ldap_server, use_ssl=True)
        user_dn = f"{settings.ldap_domain}\\{username}"

        conn = ldap3.Connection(server, user=user_dn, password=password, auto_bind=True)

        # Search for user attributes (escaped to prevent LDAP injection)
        safe_username = _ldap_escape(username)
        conn.search(
            settings.ldap_base_dn,
            f"(sAMAccountName={safe_username})",
            attributes=["displayName", "mail", "memberOf"],
        )

        if not conn.entries:
            logger.warning("LDAP auth: user '%s' bound successfully but no entry found.", username)
            return None

        entry = conn.entries[0]
        groups = [str(g) for g in entry.memberOf.values] if hasattr(entry, "memberOf") else []

        logger.info("LDAP auth: user '%s' authenticated successfully.", username)
        return {
            "username": username,
            "display_name": str(entry.displayName) if hasattr(entry, "displayName") else username,
            "email": str(entry.mail) if hasattr(entry, "mail") else None,
            "groups": groups,
        }
    except Exception:
        logger.exception("LDAP auth failed for user '%s'.", username)
        return None


def resolve_role(groups: list[str]) -> str:
    """Map AD group membership to application role."""
    group_names = {g.split(",")[0].replace("CN=", "") for g in groups}

    for role, ad_group in settings.role_group_map.items():
        if ad_group in group_names:
            return role

    return "pm"  # Default role
