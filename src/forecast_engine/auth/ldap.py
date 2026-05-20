"""LDAP / Active Directory authentication."""

from __future__ import annotations

from forecast_engine.config import settings


async def ldap_authenticate(username: str, password: str) -> dict | None:
    """Bind to AD and return user info dict, or None on failure.

    Returns: {"username": ..., "display_name": ..., "email": ..., "groups": [...]}
    """
    if not settings.ldap_server:
        return None

    try:
        import ldap3
        server = ldap3.Server(settings.ldap_server, use_ssl=True)
        user_dn = f"{settings.ldap_domain}\\{username}"

        conn = ldap3.Connection(server, user=user_dn, password=password, auto_bind=True)

        # Search for user attributes
        conn.search(
            settings.ldap_base_dn,
            f"(sAMAccountName={username})",
            attributes=["displayName", "mail", "memberOf"],
        )

        if not conn.entries:
            return None

        entry = conn.entries[0]
        groups = [str(g) for g in entry.memberOf.values] if hasattr(entry, "memberOf") else []

        return {
            "username": username,
            "display_name": str(entry.displayName) if hasattr(entry, "displayName") else username,
            "email": str(entry.mail) if hasattr(entry, "mail") else None,
            "groups": groups,
        }
    except Exception:
        return None


def resolve_role(groups: list[str]) -> str:
    """Map AD group membership to application role."""
    group_names = {g.split(",")[0].replace("CN=", "") for g in groups}

    for role, ad_group in settings.role_group_map.items():
        if ad_group in group_names:
            return role

    return "pm"  # Default role
