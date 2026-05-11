from copy import deepcopy
from typing import Annotated, TypeAlias

from pydantic import BaseModel, Field

from llmai.shared.errors import configuration_error

SchemaLike: TypeAlias = Annotated[
    dict | type[BaseModel] | BaseModel | None,
    Field(union_mode="left_to_right"),
]


def get_schema_as_dict(
    schema: SchemaLike,
    default: dict | None = None,
    *,
    strict: bool = False,
) -> dict:
    del strict

    if isinstance(schema, dict):
        return deepcopy(schema)

    if isinstance(schema, BaseModel):
        return _model_json_schema(schema.__class__)

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return _model_json_schema(schema)

    if default is None:
        return {}

    return deepcopy(default)


def strip_schema_keys(
    schema: dict,
    *,
    keys: set[str],
) -> dict:
    return _strip_schema_keys(deepcopy(schema), keys=keys)


def _strip_schema_keys(
    schema: object,
    *,
    keys: set[str],
) -> object:
    if isinstance(schema, dict):
        return {
            key: _strip_schema_keys(value, keys=keys)
            for key, value in schema.items()
            if key not in keys
        }

    if isinstance(schema, list):
        return [_strip_schema_keys(each, keys=keys) for each in schema]

    return schema


def process_schema(
    schema: dict,
    *,
    flatten_refs: bool = False,
    flatten_allof: bool = False,
    collapse_anyof: bool = False,
    ensure_additional_properties: bool = False,
    forbid_additional_properties: bool = False,
    remove_additional_properties: bool = False,
    supported_string_types: list[str] | None = None,
    supported_schema_fields: list[str] | None = None,
) -> dict:
    processed = deepcopy(schema)

    if flatten_refs or flatten_allof or collapse_anyof:
        original_definitions = {
            key: deepcopy(value)
            for key, value in processed.items()
            if key in {"$defs", "definitions"}
        }
        processed = _flatten_schema(
            processed,
            root=processed,
            seen_refs=frozenset(),
            flatten_refs=flatten_refs,
            flatten_allof=flatten_allof,
            collapse_anyof=collapse_anyof,
            in_definition=False,
            in_allof=False,
        )
        if flatten_refs and _has_def_ref(processed):
            processed.update(original_definitions)

    processed = _filter_schema(
        processed,
        supported_string_types=(
            set(supported_string_types) if supported_string_types is not None else None
        ),
        supported_schema_fields=(
            set(supported_schema_fields) if supported_schema_fields is not None else None
        ),
        in_named_schema_map=False,
    )

    if ensure_additional_properties or forbid_additional_properties:
        processed = _ensure_additional_properties(processed)

    if remove_additional_properties:
        processed = _strip_schema_keys(
            processed,
            keys={"additionalProperties", "additional_properties"},
        )

    return processed


def _flatten_schema(
    schema: object,
    *,
    root: dict,
    seen_refs: frozenset[str],
    flatten_refs: bool,
    flatten_allof: bool,
    collapse_anyof: bool,
    in_definition: bool,
    in_allof: bool,
) -> object:
    if isinstance(schema, dict):
        flattened = {
            key: _flatten_schema(
                value,
                root=root,
                seen_refs=seen_refs,
                flatten_refs=flatten_refs,
                flatten_allof=flatten_allof,
                collapse_anyof=collapse_anyof,
                in_definition=key in {"$defs", "definitions"},
                in_allof=in_allof or key == "allOf",
            )
            for key, value in schema.items()
            if (
                (key != "$ref" or not (flatten_refs or in_allof))
                and (
                    not flatten_refs
                    or in_definition
                    or key not in {"$defs", "definitions"}
                )
            )
        }

        ref = schema.get("$ref")
        if (flatten_refs or in_allof) and isinstance(ref, str):
            if ref in seen_refs:
                return {**{"$ref": ref}, **flattened}

            resolved = _resolve_local_ref(root, ref)
            if isinstance(resolved, dict):
                flattened = {
                    **_flatten_schema(
                        resolved,
                        root=root,
                        seen_refs=seen_refs | {ref},
                        flatten_refs=flatten_refs,
                        flatten_allof=flatten_allof,
                        collapse_anyof=collapse_anyof,
                        in_definition=in_definition,
                        in_allof=in_allof,
                    ),
                    **flattened,
                }

        if flatten_allof:
            flattened = _merge_allof(flattened)

        if collapse_anyof:
            flattened = _collapse_anyof(flattened)

        return flattened

    if isinstance(schema, list):
        return [
            _flatten_schema(
                each,
                root=root,
                seen_refs=seen_refs,
                flatten_refs=flatten_refs,
                flatten_allof=flatten_allof,
                collapse_anyof=collapse_anyof,
                in_definition=in_definition,
                in_allof=in_allof,
            )
            for each in schema
        ]

    return schema


def _merge_allof(schema: dict) -> dict:
    if isinstance(schema.get("allOf"), list):
        return _merge_subschema_keyword(schema, "allOf")

    return schema


def _collapse_anyof(schema: dict) -> dict:
    subschemas = schema.get("anyOf")
    if not isinstance(subschemas, list) or not subschemas:
        return schema

    if not all(isinstance(item, dict) for item in subschemas):
        return schema

    schema_types = [item.get("type") for item in subschemas]
    if not all(isinstance(item_type, str) for item_type in schema_types):
        return schema

    common_type = schema_types[0]
    if any(item_type != common_type for item_type in schema_types):
        return schema

    collapsed: dict = {"type": common_type}
    enums: list[object] = []
    all_have_enum = True
    for item in subschemas:
        item_enum = item.get("enum")
        if not isinstance(item_enum, list):
            all_have_enum = False
            continue

        for value in item_enum:
            if value not in enums:
                enums.append(value)

    if all_have_enum and enums:
        collapsed["enum"] = enums

    siblings = {key: value for key, value in schema.items() if key != "anyOf"}
    return _merge_schema_dicts(collapsed, siblings)


def _merge_subschema_keyword(schema: dict, key: str) -> dict:
    subschemas = schema.get(key)
    if not isinstance(subschemas, list):
        return schema

    merged: dict = {}
    for item in subschemas:
        if isinstance(item, dict):
            merged = _merge_schema_dicts(merged, item)
        else:
            return schema

    siblings = {each_key: value for each_key, value in schema.items() if each_key != key}
    return _merge_schema_dicts(merged, siblings)


def _merge_schema_dicts(left: dict, right: dict) -> dict:
    merged = deepcopy(left)

    for key, value in right.items():
        if (
            key == "properties"
            and isinstance(merged.get(key), dict)
            and isinstance(value, dict)
        ):
            merged[key] = {**merged[key], **value}
            continue

        if (
            key == "required"
            and isinstance(merged.get(key), list)
            and isinstance(value, list)
        ):
            merged[key] = [
                *merged[key],
                *[item for item in value if item not in merged[key]],
            ]
            continue

        merged[key] = value

    return merged


def _resolve_local_ref(root: dict, ref: str) -> object:
    if ref == "#":
        return root

    if not ref.startswith("#/"):
        return {"$ref": ref}

    current: object = root
    for part in ref[2:].split("/"):
        if not isinstance(current, dict):
            return {"$ref": ref}

        key = part.replace("~1", "/").replace("~0", "~")
        if key not in current:
            return {"$ref": ref}

        current = current[key]

    return current


def _has_def_ref(schema: object) -> bool:
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str) and (
            ref.startswith("#/$defs/") or ref.startswith("#/definitions/")
        ):
            return True

        return any(_has_def_ref(value) for value in schema.values())

    if isinstance(schema, list):
        return any(_has_def_ref(item) for item in schema)

    return False


def _filter_schema(
    schema: object,
    *,
    supported_string_types: set[str] | None,
    supported_schema_fields: set[str] | None,
    in_named_schema_map: bool,
) -> object:
    if isinstance(schema, list):
        return [
            _filter_schema(
                item,
                supported_string_types=supported_string_types,
                supported_schema_fields=supported_schema_fields,
                in_named_schema_map=False,
            )
            for item in schema
        ]

    if not isinstance(schema, dict):
        return schema

    filtered: dict = {}
    is_string_schema = schema.get("type") == "string"

    for key, value in schema.items():
        if _should_drop_schema_key(
            key,
            value,
            is_string_schema=is_string_schema,
            supported_string_types=supported_string_types,
            supported_schema_fields=supported_schema_fields,
            in_named_schema_map=in_named_schema_map,
        ):
            continue

        filtered[key] = _filter_schema(
            value,
            supported_string_types=supported_string_types,
            supported_schema_fields=supported_schema_fields,
            in_named_schema_map=(
                False
                if in_named_schema_map
                else key in {"properties", "$defs", "definitions"}
            ),
        )

    return filtered


def _ensure_additional_properties(schema: object) -> object:
    if isinstance(schema, list):
        return [_ensure_additional_properties(item) for item in schema]

    if not isinstance(schema, dict):
        return schema

    ensured = {
        key: _ensure_additional_properties(value)
        for key, value in schema.items()
    }
    if ensured.get("type") == "object":
        ensured["additionalProperties"] = False

    return ensured


def _should_drop_schema_key(
    key: str,
    value: object,
    *,
    is_string_schema: bool,
    supported_string_types: set[str] | None,
    supported_schema_fields: set[str] | None,
    in_named_schema_map: bool,
) -> bool:
    if (
        is_string_schema
        and key == "format"
        and supported_string_types is not None
        and isinstance(value, str)
        and value not in supported_string_types
    ):
        return True

    if supported_schema_fields is None or in_named_schema_map:
        return False

    return key not in supported_schema_fields


def _model_json_schema(model_type: type[BaseModel]) -> dict:
    if model_type is BaseModel:
        raise configuration_error(
            "Schema must be a dict, a BaseModel subclass, or an instance of a BaseModel subclass",
            status_code=400,
        )

    return model_type.model_json_schema()
