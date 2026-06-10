import re


def _canon_field_name(text):
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text or ""))
    value = value.replace("_", " ")
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value).strip().lower()
    replacements = {
        "document": "doc",
        "organisation": "organization",
    }
    tokens = [replacements.get(tok, tok) for tok in value.split() if tok]
    return " ".join(tokens)


def _split_helper_suffix(field_name):
    name = str(field_name or "").strip()
    if name.endswith("_textMode"):
        return name[:-9], "_textMode"
    if name.endswith("_"):
        return name[:-1], "_"
    return name, ""


def _resolve_runtime_field_name(field_name, runtime_field_names):
    name = str(field_name or "").strip()
    if not name:
        return name
    runtime_names = [str(item or "").strip() for item in (runtime_field_names or []) if str(item or "").strip()]
    if not runtime_names:
        return name
    base_name, suffix = _split_helper_suffix(name)
    canon_target = _canon_field_name(base_name)
    if not canon_target:
        return name

    matches = []
    for runtime in runtime_names:
        runtime_base, runtime_suffix = _split_helper_suffix(runtime)
        if suffix and runtime_suffix != suffix:
            continue
        if _canon_field_name(runtime_base) == canon_target:
            matches.append(runtime)
    if matches:
        def _match_rank(runtime_name):
            runtime_base, runtime_suffix = _split_helper_suffix(runtime_name)
            exact = 1 if runtime_name == name else 0
            return (len(runtime_base), -exact, len(runtime_name))

        matches = sorted(set(matches), key=_match_rank)
        return matches[0]

    if suffix:
        base_matches = []
        for runtime in runtime_names:
            runtime_base, runtime_suffix = _split_helper_suffix(runtime)
            if runtime_suffix:
                continue
            if _canon_field_name(runtime_base) == canon_target:
                base_matches.append(runtime_base)
        if base_matches:
            candidate = f"{base_matches[0]}{suffix}"
            if candidate in runtime_set:
                return candidate
            return base_matches[0]
    runtime_set = set(runtime_names)
    if name in runtime_set:
        return name
    return name


def _remap_value_to_runtime(value, runtime_field_names):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            key_text = str(key)
            mapped_key = key_text if key_text.startswith("$") else _resolve_runtime_field_name(key_text, runtime_field_names)
            out[mapped_key] = _remap_value_to_runtime(child, runtime_field_names)
        return out
    if isinstance(value, list):
        return [_remap_value_to_runtime(item, runtime_field_names) for item in value]
    if isinstance(value, str) and value.startswith("$") and not value.startswith("$$"):
        body = value[1:]
        if "." in body:
            root, rest = body.split(".", 1)
            mapped_root = _resolve_runtime_field_name(root, runtime_field_names)
            return f"${mapped_root}.{rest}"
        return f"${_resolve_runtime_field_name(body, runtime_field_names)}"
    return value


def _remap_plan_to_runtime_fields(plan, runtime_field_names):
    if not isinstance(plan, dict) or not runtime_field_names:
        return plan
    mapped = dict(plan)
    if "filter" in mapped:
        mapped["filter"] = _remap_value_to_runtime(mapped.get("filter"), runtime_field_names)
    if "projection" in mapped:
        mapped["projection"] = _remap_value_to_runtime(mapped.get("projection"), runtime_field_names)
    if "pipeline" in mapped:
        mapped["pipeline"] = _remap_value_to_runtime(mapped.get("pipeline"), runtime_field_names)
    sort_value = mapped.get("sort")
    if isinstance(sort_value, list):
        remapped_sort = []
        for item in sort_value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                remapped_sort.append([
                    _resolve_runtime_field_name(item[0], runtime_field_names),
                    item[1],
                ])
            else:
                remapped_sort.append(item)
        mapped["sort"] = remapped_sort
    return mapped


def remap_plan_runtime_fields(plan, runtime_field_names):
    return _remap_plan_to_runtime_fields(plan, runtime_field_names)
