from erp_backend.core.utils import normalize_lookup_text
from erp_backend.storage.mongo import collection_is_allowed


def is_user_permission_request(user_input):
    text = normalize_lookup_text(user_input)
    tokens = set(text.split())
    user_words = {"user", "users", "username", "usernames"}
    permission_words = {"permission", "permissions", "access", "allowed", "role", "roles", "rbac"}
    return bool(tokens & user_words) and bool(tokens & permission_words)


def can_view_user_permissions(accessible_collections, selected_user):
    allowed = selected_user.get("allowed_collections", [])
    return (
        "*" in allowed
        or "users" in accessible_collections
        or "user" in accessible_collections
        or collection_is_allowed("users", allowed)
        or collection_is_allowed("user", allowed)
    )


def permission_table_label(collection_name, table_metadata):
    return table_metadata.get(collection_name, {}).get("template_name", collection_name)


def user_permission_rows(rbac_users, collections, table_metadata):
    rows = []
    for user in rbac_users:
        allowed = user.get("allowed_collections") or []
        if "*" in allowed:
            permissions = "All ERP tables"
            allowed_count = len(collections)
        else:
            visible_allowed = [name for name in collections if name in allowed]
            permissions = ", ".join(
                permission_table_label(name, table_metadata)
                for name in visible_allowed
            ) or "No table access"
            allowed_count = len(visible_allowed)

        rows.append(
            {
                "User Name": user.get("display_name") or user.get("user_id") or "",
                "Email": user.get("email") or "",
                "Roles": ", ".join(user.get("roles") or []) or "Not assigned",
                "Allowed Table Count": allowed_count,
                "Permissions": permissions,
            }
        )
    return rows


def user_permissions_answer(rows):
    if not rows:
        return "I did not find any RBAC users."
    return (
        f"I found {len(rows)} users and listed their visible table permissions. "
        "The table shows each user, assigned roles when available, and the ERP tables they can access."
    )
