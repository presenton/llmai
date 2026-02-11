from collections.abc import Mapping, Sequence
from copy import deepcopy
from logging import Logger
import re
from typing import Any, List, Optional

from jsonschema import ValidationError, validate

from utils.dict_utils import (
    get_dict_paths_with_key,
    get_dict_at_path,
)

supported_string_formats = [
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "ipv4",
    "ipv6",
    "uuid",
]


def remove_fields_from_schema(schema: dict, fields_to_remove: List[str]):
    schema = deepcopy(schema)
    properties_paths = get_dict_paths_with_key(schema, "properties")
    for path in properties_paths:
        parent_obj = get_dict_at_path(schema, path)
        if "properties" in parent_obj and isinstance(parent_obj["properties"], dict):
            for field in fields_to_remove:
                if field in parent_obj["properties"]:
                    del parent_obj["properties"][field]

    required_paths = get_dict_paths_with_key(schema, "required")
    for path in required_paths:
        parent_obj = get_dict_at_path(schema, path)
        if "required" in parent_obj and isinstance(parent_obj["required"], list):
            parent_obj["required"] = [
                field
                for field in parent_obj["required"]
                if field not in fields_to_remove
            ]

    return schema


def add_field_in_schema(schema: dict, field: dict, required: bool = False) -> dict:
    if not isinstance(field, dict) or len(field) != 1:
        raise ValueError(
            "`field` must be a dict with exactly one entry: {name: schema_dict}"
        )

    field_name, field_schema = next(iter(field.items()))
    if not isinstance(field_name, str):
        raise TypeError("Field name must be a string")
    if not isinstance(field_schema, dict):
        raise TypeError("Field schema must be a dictionary")

    updated_schema: dict = deepcopy(schema)

    root_properties = updated_schema.get("properties")
    if not isinstance(root_properties, dict):
        updated_schema["properties"] = {}
        root_properties = updated_schema["properties"]

    root_properties[field_name] = field_schema

    # Update root-level required based on the flag
    existing_required = updated_schema.get("required")
    if not isinstance(existing_required, list):
        existing_required = []

    if required:
        if field_name not in existing_required:
            existing_required.append(field_name)
    else:
        if field_name in existing_required:
            existing_required = [
                name for name in existing_required if name != field_name
            ]

    if existing_required:
        updated_schema["required"] = existing_required
    else:
        updated_schema.pop("required", None)

    return updated_schema


# From OpenAI
def ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    if not isinstance(json_schema, dict):
        raise TypeError(f"Expected dict; path={path}")

    schema = deepcopy(json_schema)
    if root is json_schema:
        root = schema

    # ---- defs / definitions ----
    for defs_key in ("$defs", "definitions"):
        defs = schema.get(defs_key)
        if isinstance(defs, dict):
            for name, def_schema in defs.items():
                defs[name] = ensure_strict_json_schema(
                    def_schema, path=(*path, defs_key, name), root=root
                )

    # ---- strip forbidden keys ----
    schema.pop("default", None)
    schema.pop("examples", None)

    # ---- detect object schemas (structural, not just type-based) ----
    is_object = (
        schema.get("type") == "object"
        or "properties" in schema
        or "additionalProperties" in schema
    )

    if is_object:
        schema["type"] = "object"
        schema["additionalProperties"] = False

    # ---- properties ----
    properties = schema.get("properties")
    if isinstance(properties, dict):
        schema["properties"] = {
            key: ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }
        # DO NOT auto-generate `required`

    # ---- arrays ----
    items = schema.get("items")
    if isinstance(items, dict):
        schema["items"] = ensure_strict_json_schema(
            items, path=(*path, "items"), root=root
        )

    # ---- unions ----
    for union_key in ("anyOf", "oneOf", "allOf"):
        variants = schema.get(union_key)
        if isinstance(variants, list):
            schema[union_key] = [
                ensure_strict_json_schema(v, path=(*path, union_key, str(i)), root=root)
                for i, v in enumerate(variants)
            ]

    # ---- string formats ----
    if schema.get("type") == "string" and "format" in schema:
        if schema["format"] not in supported_string_formats:
            del schema["format"]

    # ---- $ref expansion ----
    ref = schema.get("$ref")
    if ref and len(schema) > 1:
        resolved = resolve_ref(root=root, ref=ref)
        if not isinstance(resolved, dict):
            raise ValueError(f"Invalid $ref target at {path}")

        merged = {**resolved, **schema}
        merged.pop("$ref", None)
        return ensure_strict_json_schema(merged, path=path, root=root)

    return schema


def ensure_more_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
    remove_min_length: bool = True,
    remove_max_length: bool = True,
    remove_min_items: bool = True,
    remove_max_items: bool = True,
    remove_maximum: bool = True,
    remove_minimum: bool = True,
    remove_max_words: bool = True,
    remove_min_words: bool = True,
) -> dict[str, Any]:
    """Wrapper around `ensure_strict_json_schema` that optionally removes length constraints."""

    schema = ensure_strict_json_schema(json_schema, path=path, root=root)

    def _strip_length_constraints(node: Any) -> None:
        if isinstance(node, dict):
            node_type = node.get("type")
            if node_type == "string":
                if remove_min_length:
                    node.pop("minLength", None)
                if remove_max_length:
                    node.pop("maxLength", None)
                if remove_min_words:
                    node.pop("minWords", None)
                if remove_max_words:
                    node.pop("maxWords", None)
            if node_type == "array":
                if remove_min_items:
                    node.pop("minItems", None)
                if remove_max_items:
                    node.pop("maxItems", None)
            if node_type == "number":
                if remove_maximum:
                    node.pop("maximum", None)
                if remove_minimum:
                    node.pop("minimum", None)
            for value in node.values():
                _strip_length_constraints(value)
        elif isinstance(node, list):
            for entry in node:
                _strip_length_constraints(entry)

    _strip_length_constraints(schema)

    # Also remove maxWords and minWords from top level if they exist
    if remove_max_words:
        schema.pop("maxWords", None)
    if remove_min_words:
        schema.pop("minWords", None)

    return schema


def set_value_at_path(container: Any, path: list[Any], value: Any) -> None:
    """Set a value within a nested dict/list structure given a traversal path."""
    if not path:
        raise ValueError("Path must contain at least one element")

    current = container
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def clip_array_max_items_violation(
    content: dict, error: ValidationError, logger: Optional[Logger] = None
) -> bool:
    """Trim arrays that violate `maxItems` constraints."""
    max_items = error.schema.get("maxItems")
    path = list(error.path)
    if (
        error.validator != "maxItems"
        or not isinstance(error.instance, list)
        or not isinstance(max_items, int)
        or not path
    ):
        return False

    clipped_value = error.instance[:max_items]
    set_value_at_path(content, path, clipped_value)

    if logger:
        logger.info(
            "Clipped array at path %s to maxItems %d (was %d items)",
            " -> ".join(str(part) for part in path),
            max_items,
            len(error.instance),
        )
    return True


def _coerce_model_like(value: Any) -> Any:
    """Convert SDK-specific objects (e.g. OpenAI AttributedDict) into primitive types."""
    if isinstance(value, (str, bytes, bytearray)):
        return value

    for attr in ("model_dump", "dict", "to_dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                continue
    return value


def _normalize_json_like(value: Any) -> Any:
    """Recursively coerce SDK/container objects into plain Python collections."""
    coerced_value = _coerce_model_like(value)

    if isinstance(coerced_value, Mapping):
        return {key: _normalize_json_like(val) for key, val in coerced_value.items()}

    if isinstance(coerced_value, Sequence) and not isinstance(
        coerced_value, (str, bytes, bytearray)
    ):
        return [_normalize_json_like(item) for item in coerced_value]

    return coerced_value


def _ensure_dict_content(content: Any) -> dict:
    normalized = _normalize_json_like(content)
    if not isinstance(normalized, dict):
        raise TypeError("Structured LLM response must be a dictionary")
    return normalized


def _collect_max_items_errors(error: ValidationError) -> list[ValidationError]:
    errors = []
    if error.validator == "maxItems":
        errors.append(error)
    for suberror in error.context:
        errors.extend(_collect_max_items_errors(suberror))
    return errors


def _clip_all_max_items_violations(
    content: dict,
    errors: list[ValidationError],
    logger: Optional[Logger] = None,
) -> bool:
    fixed = False

    for error in errors:
        success = clip_array_max_items_violation(content, error, logger)
        fixed = fixed or success

    return fixed


def verify_content_against_schema(
    content: Any,
    schema: dict,
    should_raise: bool = True,
    logger: Optional[Logger] = None,
) -> dict:
    normalized_content = _ensure_dict_content(content)
    content_copy = deepcopy(normalized_content)

    while True:
        try:
            validate(instance=content_copy, schema=schema)
            return content_copy
        except ValidationError as error:
            if logger:
                logger.info("Validation error, attempting auto-fix")

            max_item_errors = _collect_max_items_errors(error)
            if should_raise and not max_item_errors:
                raise

            fixed = _clip_all_max_items_violations(
                content_copy, max_item_errors, logger
            )

            if should_raise and not fixed:
                raise
            else:
                return content_copy


def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert isinstance(value, dict), (
            f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        )
        resolved = value

    return resolved


# Flattens a JSON schema by inlining all $ref references and removing $defs/definitions
def flatten_json_schema(schema: dict) -> dict:
    root_schema = deepcopy(schema)

    def _flatten(node: Any) -> Any:
        if isinstance(node, dict):
            # If node is a pure $ref (or combined with extra fields), inline it
            if "$ref" in node:
                ref_value = node["$ref"]
                assert isinstance(ref_value, str), (
                    f"Received non-string $ref - {ref_value}"
                )
                resolved = resolve_ref(root=root_schema, ref=ref_value)
                assert isinstance(resolved, dict), (
                    f"Expected `$ref: {ref_value}` to resolve to a dictionary but got {type(resolved)}"
                )
                # Merge: referenced first, then overlay current (excluding $ref)
                merged: dict[str, Any] = deepcopy(resolved)
                for key, value in node.items():
                    if key == "$ref":
                        continue
                    merged[key] = value
                return _flatten(merged)

            flattened: dict[str, Any] = {}
            for key, value in node.items():
                # Drop defs/definitions in output
                if key in ("$defs", "definitions"):
                    continue
                if key == "properties" and isinstance(value, dict):
                    flattened[key] = {
                        prop_key: _flatten(prop_val)
                        for prop_key, prop_val in value.items()
                    }
                elif key in ("items", "contains", "additionalProperties", "not"):
                    if isinstance(value, dict):
                        flattened[key] = _flatten(value)
                    elif isinstance(value, list):
                        flattened[key] = [_flatten(v) for v in value]
                    else:
                        flattened[key] = value
                elif key in ("allOf", "anyOf", "oneOf", "prefixItems") and isinstance(
                    value, list
                ):
                    flattened[key] = [_flatten(v) for v in value]
                else:
                    flattened[key] = (
                        _flatten(value) if isinstance(value, (dict, list)) else value
                    )
            return flattened
        if isinstance(node, list):
            return [_flatten(v) for v in node]
        return node

    result = _flatten(schema)
    # Ensure top-level cleanup just in case
    if isinstance(result, dict):
        result.pop("$defs", None)
        result.pop("definitions", None)
    return result


def remove_titles_from_schema(schema: dict) -> dict[str, Any]:
    def _strip_titles(node: Any) -> Any:
        if isinstance(node, dict):
            rebuilt: dict[str, Any] = {}
            for key, value in node.items():
                # Preserve properties named "title" under the JSON Schema "properties" mapping
                if key == "properties" and isinstance(value, dict):
                    rebuilt[key] = {
                        prop_name: _strip_titles(prop_schema)
                        for prop_name, prop_schema in value.items()
                    }
                    continue

                # Remove schema metadata field "title" elsewhere
                if key == "title":
                    continue

                rebuilt[key] = _strip_titles(value)
            return rebuilt
        if isinstance(node, list):
            return [_strip_titles(item) for item in node]
        return node

    return _strip_titles(deepcopy(schema))


def remove_defaults_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of `schema` with all JSON Schema `default` entries removed.
    """

    def _strip_defaults(node: Any) -> Any:
        if isinstance(node, dict):
            rebuilt: dict[str, Any] = {}
            for key, value in node.items():
                # Recurse into properties to keep structure intact
                if key == "properties" and isinstance(value, dict):
                    rebuilt[key] = {
                        prop_name: _strip_defaults(prop_schema)
                        for prop_name, prop_schema in value.items()
                    }
                    continue

                if key == "default":
                    continue

                rebuilt[key] = _strip_defaults(value)
            return rebuilt

        if isinstance(node, list):
            return [_strip_defaults(item) for item in node]

        return node

    return _strip_defaults(deepcopy(schema))


def remove_descriptions_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Return a copy of `schema` with all JSON Schema `description` entries removed.
    """

    def _strip_descriptions(node: Any) -> Any:
        if isinstance(node, dict):
            rebuilt: dict[str, Any] = {}
            for key, value in node.items():
                # Recurse into properties to keep structure intact
                if key == "properties" and isinstance(value, dict):
                    rebuilt[key] = {
                        prop_name: _strip_descriptions(prop_schema)
                        for prop_name, prop_schema in value.items()
                    }
                    continue

                if key == "description":
                    continue

                rebuilt[key] = _strip_descriptions(value)
            return rebuilt

        if isinstance(node, list):
            return [_strip_descriptions(item) for item in node]

        return node

    return _strip_descriptions(deepcopy(schema))


def get_schema_markdown(
    schema: dict,
    *,
    include_name: bool = False,
    include_chart_fields: bool = False,
    include_table_fields: bool = False,
    include_image_fields: bool = False,
    include_icon_fields: bool = False,
) -> str:
    """
    Build a concise markdown description of schema properties.
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        return ""

    def _collect_constraints(field_def: dict) -> list[str]:
        constraints: list[str] = []
        const_value = field_def.get("const")
        if const_value is not None:
            if isinstance(const_value, str):
                constraints.append(f'const "{const_value}"')
            else:
                constraints.append(f"const {const_value}")
        max_length = field_def.get("maxLength")
        if isinstance(max_length, int):
            constraints.append(f"max {max_length} characters")
        min_length = field_def.get("minLength")
        if isinstance(min_length, int):
            constraints.append(f"min {min_length} characters")
        max_items = field_def.get("maxItems")
        if isinstance(max_items, int):
            constraints.append(f"max {max_items} items")
        min_items = field_def.get("minItems")
        if isinstance(min_items, int):
            constraints.append(f"min {min_items} items")
        return constraints

    def _format_constraints(field_def: dict) -> str:
        constraints = _collect_constraints(field_def)
        if not constraints:
            return ""
        return f" ({', '.join(constraints)})"

    def _is_image_object(field_def: dict) -> bool:
        properties = field_def.get("properties")
        if not isinstance(properties, dict):
            return False
        return "__image_url__" in properties and "__image_prompt__" in properties

    def _is_icon_object(field_def: dict) -> bool:
        properties = field_def.get("properties")
        if not isinstance(properties, dict):
            return False
        return "__icon_url__" in properties and "__icon_query__" in properties

    def _get_media_label(field_def: dict) -> str | None:
        if _is_icon_object(field_def):
            return "icon"
        if _is_image_object(field_def):
            return "image"
        if field_def.get("type") == "array":
            items_def = field_def.get("items")
            if isinstance(items_def, dict):
                if _is_icon_object(items_def):
                    return "icon"
                if _is_image_object(items_def):
                    return "image"
            if isinstance(items_def, list):
                for item_def in items_def:
                    if not isinstance(item_def, dict):
                        continue
                    if _is_icon_object(item_def):
                        return "icon"
                    if _is_image_object(item_def):
                        return "image"
        return None

    def _get_type_label(field_def: dict) -> str:
        media_label = _get_media_label(field_def)
        if media_label:
            return media_label
        any_of = field_def.get("anyOf")
        if isinstance(any_of, list) and any_of:
            return "anyOf"
        one_of = field_def.get("oneOf")
        if isinstance(one_of, list) and one_of:
            return "oneOf"
        all_of = field_def.get("allOf")
        if isinstance(all_of, list) and all_of:
            return "allOf"
        field_type = field_def.get("type")
        if isinstance(field_type, str):
            return field_type
        if "properties" in field_def:
            return "object"
        if "items" in field_def:
            return "array"
        return "value"

    def _get_required_list(obj_def: dict) -> list[str]:
        required = obj_def.get("required")
        return required if isinstance(required, list) else []

    def _field_name_tokens(field_name: str) -> list[str]:
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", field_name)
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized)
        return [token for token in normalized.lower().split("_") if token]

    def _get_chart_graph_label(field_name: str) -> str | None:
        tokens = _field_name_tokens(field_name)
        if "charts" in tokens or any(token.startswith("charts") for token in tokens):
            return "chart"
        if "chart" in tokens or any(token.startswith("chart") for token in tokens):
            return "chart"
        if "graphs" in tokens or any(token.startswith("graphs") for token in tokens):
            return "graph"
        if "graph" in tokens or any(token.startswith("graph") for token in tokens):
            return "graph"
        return None

    def _get_table_label(field_name: str) -> str | None:
        tokens = _field_name_tokens(field_name)
        if "tables" in tokens or any(token.startswith("tables") for token in tokens):
            return "table"
        if "table" in tokens or any(token.startswith("table") for token in tokens):
            return "table"
        return None

    def _pluralize_chart_table_label(label: str, field_def: dict) -> str:
        if label not in {"chart", "graph", "table"}:
            return label
        field_type = field_def.get("type")
        if field_type == "array" or "items" in field_def:
            return f"{label}s"
        return label

    def _is_suppressed_field(field_name: str, field_def: dict) -> bool:
        chart_graph_label = _get_chart_graph_label(field_name)
        if chart_graph_label and not include_chart_fields:
            return True
        table_label = _get_table_label(field_name)
        if table_label and not include_table_fields:
            return True
        media_label = _get_media_label(field_def)
        if media_label == "image" and not include_image_fields:
            return True
        if media_label == "icon" and not include_icon_fields:
            return True
        return False

    def _format_field_line(
        *,
        field_name: str,
        label: str,
        media_label: str | None,
        constraint_text: str,
        optional_suffix: str,
        suppressed: bool,
    ) -> str:
        if suppressed:
            if media_label in {"image", "icon"}:
                if include_name:
                    return f"{field_name}: {label}{constraint_text}{optional_suffix}"
                return f"{label}{constraint_text}{optional_suffix}"
            if include_name:
                return f"{field_name}: {label}{constraint_text}{optional_suffix}"
            if label in {"chart", "charts", "graph", "graphs", "table", "tables"}:
                return f"{label}{constraint_text}{optional_suffix}"
            display_name = field_name
            return f"{display_name}{constraint_text}{optional_suffix}"
        if include_name:
            return f"{field_name}: {label}{constraint_text}{optional_suffix}"
        return f"{label}{constraint_text}{optional_suffix}"

    def _add_nested_fields(
        lines_out: list[str],
        field_def: dict,
        indent_level: int,
    ) -> None:
        media_label = _get_media_label(field_def)
        if media_label == "image" and not include_image_fields:
            return
        if media_label == "icon" and not include_icon_fields:
            return
        nested_props = field_def.get("properties")
        if not isinstance(nested_props, dict) or not nested_props:
            return
        required_fields = set(_get_required_list(field_def))
        nested_index = 1
        for nested_name, nested_def in nested_props.items():
            if not isinstance(nested_def, dict):
                continue
            indent = "  " * indent_level
            suppressed = _is_suppressed_field(nested_name, nested_def)
            type_label = _get_type_label(nested_def)
            media_label = _get_media_label(nested_def)
            chart_graph_label = _get_chart_graph_label(nested_name)
            table_label = _get_table_label(nested_name)
            label = (
                _pluralize_chart_table_label(chart_graph_label, nested_def)
                if chart_graph_label
                else _pluralize_chart_table_label(table_label, nested_def)
                if table_label
                else type_label
            )
            constraint_text = _format_constraints(nested_def)
            optional_suffix = "" if nested_name in required_fields else " (optional)"
            lines_out.append(
                f"{indent}{nested_index}. "
                f"{_format_field_line(field_name=nested_name, label=label, media_label=media_label, constraint_text=constraint_text, optional_suffix=optional_suffix, suppressed=suppressed)}"
            )
            if suppressed:
                nested_index += 1
                continue
            if chart_graph_label and not include_chart_fields:
                nested_index += 1
                continue
            if table_label and not include_table_fields:
                nested_index += 1
                continue
            if media_label == "image" and not include_image_fields:
                nested_index += 1
                continue
            if media_label == "icon" and not include_icon_fields:
                nested_index += 1
                continue
            _add_array_items_line(lines_out, nested_def, indent_level + 1)
            _add_union_lines(lines_out, nested_def, indent_level + 1, "anyOf")
            _add_union_lines(lines_out, nested_def, indent_level + 1, "oneOf")
            _add_union_lines(lines_out, nested_def, indent_level + 1, "allOf")
            if type_label == "object" or media_label in {"image", "icon"}:
                _add_nested_fields(lines_out, nested_def, indent_level + 1)
            nested_index += 1

    def _add_union_lines(
        lines_out: list[str],
        field_def: dict,
        indent_level: int,
        union_key: str,
    ) -> None:
        variants = field_def.get(union_key)
        if not isinstance(variants, list) or not variants:
            return
        indent = "  " * indent_level
        for index, option_def in enumerate(variants, start=0):
            if not isinstance(option_def, dict):
                continue
            option_label = _get_type_label(option_def)
            option_media_label = _get_media_label(option_def)
            if option_media_label == "image" and not include_image_fields:
                continue
            if option_media_label == "icon" and not include_icon_fields:
                continue
            constraint_text = _format_constraints(option_def)
            lines_out.append(
                f"{indent}- {union_key}[{index}]: {option_label}{constraint_text}"
            )
            _add_array_items_line(lines_out, option_def, indent_level + 1)
            if option_label == "object" or option_media_label in {"image", "icon"}:
                _add_nested_fields(lines_out, option_def, indent_level + 1)

    def _add_array_items_line(
        lines_out: list[str],
        field_def: dict,
        indent_level: int,
    ) -> None:
        if field_def.get("type") != "array":
            return
        items_def = field_def.get("items")
        indent = "  " * indent_level
        if isinstance(items_def, list):
            for index, item_def in enumerate(items_def, start=1):
                if not isinstance(item_def, dict):
                    continue
                item_type_label = _get_type_label(item_def)
                media_label = _get_media_label(item_def)
                if media_label == "image" and include_image_fields:
                    lines_out.append(f"{indent}- items[{index}]: image")
                    _add_nested_fields(lines_out, item_def, indent_level + 1)
                    continue
                if media_label == "icon" and include_icon_fields:
                    lines_out.append(f"{indent}- items[{index}]: icon")
                    _add_nested_fields(lines_out, item_def, indent_level + 1)
                    continue
                if item_type_label == "object":
                    lines_out.append(f"{indent}- items[{index}]:")
                    _add_nested_fields(lines_out, item_def, indent_level + 1)
                    continue
                constraint_text = _format_constraints(item_def)
                lines_out.append(
                    f"{indent}- items[{index}]: {item_type_label}{constraint_text}"
                )
            return
        if not isinstance(items_def, dict):
            return
        item_type_label = _get_type_label(items_def)
        media_label = _get_media_label(items_def)
        if media_label == "image" and include_image_fields:
            lines_out.append(f"{indent}- items: image")
            _add_nested_fields(lines_out, items_def, indent_level + 1)
            return
        if media_label == "icon" and include_icon_fields:
            lines_out.append(f"{indent}- items: icon")
            _add_nested_fields(lines_out, items_def, indent_level + 1)
            return
        if item_type_label == "object":
            lines_out.append(f"{indent}- items:")
            _add_nested_fields(lines_out, items_def, indent_level + 1)
            return
        constraint_text = _format_constraints(items_def)
        lines_out.append(f"{indent}- items: {item_type_label}{constraint_text}")

    lines: list[str] = []
    required_fields = set(_get_required_list(schema))
    index = 1
    for prop_name, prop_def in properties.items():
        if not isinstance(prop_def, dict):
            continue
        suppressed = _is_suppressed_field(prop_name, prop_def)
        type_label = _get_type_label(prop_def)
        media_label = _get_media_label(prop_def)
        chart_graph_label = _get_chart_graph_label(prop_name)
        table_label = _get_table_label(prop_name)
        label = (
            _pluralize_chart_table_label(chart_graph_label, prop_def)
            if chart_graph_label
            else _pluralize_chart_table_label(table_label, prop_def)
            if table_label
            else type_label
        )
        constraint_text = _format_constraints(prop_def)
        optional_suffix = "" if prop_name in required_fields else " (optional)"
        lines.append(
            f"{index}. "
            f"{_format_field_line(field_name=prop_name, label=label, media_label=media_label, constraint_text=constraint_text, optional_suffix=optional_suffix, suppressed=suppressed)}"
        )
        if not suppressed:
            if chart_graph_label and not include_chart_fields:
                index += 1
                continue
            if table_label and not include_table_fields:
                index += 1
                continue
            if media_label == "image" and not include_image_fields:
                index += 1
                continue
            if media_label == "icon" and not include_icon_fields:
                index += 1
                continue
            _add_array_items_line(lines, prop_def, 1)
            _add_union_lines(lines, prop_def, 1, "anyOf")
            _add_union_lines(lines, prop_def, 1, "oneOf")
            _add_union_lines(lines, prop_def, 1, "allOf")
            if type_label == "object" or media_label in {"image", "icon"}:
                _add_nested_fields(lines, prop_def, 1)

        index += 1

    return "\n".join(lines)


# ? Not used
def generate_constraint_sentences(schema: dict) -> str:
    """
    Generate human-readable constraint sentences from a JSON schema.

    Args:
        schema: JSON schema dictionary

    Returns:
        String containing constraint sentences separated by newlines
    """
    constraints = []

    def extract_constraints_recursive(obj, prefix=""):
        if isinstance(obj, dict):
            if "properties" in obj:
                properties = obj["properties"]
                for prop_name, prop_def in properties.items():
                    current_path = f"{prefix}.{prop_name}" if prefix else prop_name

                    if isinstance(prop_def, dict):
                        prop_type = prop_def.get("type")

                        # Handle string constraints
                        if prop_type == "string":
                            min_length = prop_def.get("minLength")
                            max_length = prop_def.get("maxLength")

                            if min_length is not None and max_length is not None:
                                constraints.append(
                                    f"    - {current_path} should be less than {max_length} characters and greater than {min_length} characters"
                                )
                            elif max_length is not None:
                                constraints.append(
                                    f"    - {current_path} should be less than {max_length} characters"
                                )
                            elif min_length is not None:
                                constraints.append(
                                    f"    - {current_path} should be greater than {min_length} characters"
                                )

                        # Handle array constraints
                        elif prop_type == "array":
                            min_items = prop_def.get("minItems")
                            max_items = prop_def.get("maxItems")

                            if min_items is not None and max_items is not None:
                                constraints.append(
                                    f"    - {current_path} should have more than {min_items} items and less than {max_items} items"
                                )
                            elif max_items is not None:
                                constraints.append(
                                    f"    - {current_path} should have less than {max_items} items"
                                )
                            elif min_items is not None:
                                constraints.append(
                                    f"    - {current_path} should have more than {min_items} items"
                                )

                        # Recurse into nested objects
                        if prop_type == "object" or "properties" in prop_def:
                            extract_constraints_recursive(prop_def, current_path)

                        # Handle array items if they have properties
                        if prop_type == "array" and "items" in prop_def:
                            items_def = prop_def["items"]
                            if isinstance(items_def, dict) and (
                                "properties" in items_def
                                or items_def.get("type") == "object"
                            ):
                                extract_constraints_recursive(
                                    items_def, f"{current_path}[*]"
                                )

            # Also recurse into other nested structures
            for key, value in obj.items():
                if key not in [
                    "properties",
                    "type",
                    "minLength",
                    "maxLength",
                    "minItems",
                    "maxItems",
                ] and isinstance(value, dict):
                    extract_constraints_recursive(value, prefix)

    # Start extraction from the root schema
    extract_constraints_recursive(schema)

    return "\n".join(constraints)
