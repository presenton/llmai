import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmai.models import (
    AmbiguousModelError,
    ModelNotFoundError,
    get_context_window,
    get_model_metadata,
    list_models,
    load_model_data,
    query_models,
    refresh_model_data,
)

FIXTURE = {
    "alpha": {
        "id": "alpha",
        "name": "Alpha AI",
        "doc": "https://alpha.example",
        "models": {
            "shared": {
                "id": "shared",
                "name": "Shared Model",
                "family": "shared-family",
                "reasoning": True,
                "limit": {"context": 128_000, "output": 8_000},
            },
            "unique": {
                "id": "unique",
                "name": "Unique Model",
                "description": "A searchable description",
                "limit": {"context": 32_000, "output": 4_000},
            },
            "missing-context": {
                "id": "missing-context",
                "name": "Missing Context",
                "limit": {"output": 1_000},
            },
        },
    },
    "beta": {
        "id": "beta",
        "name": "Beta AI",
        "models": {
            "shared": {
                "id": "shared",
                "name": "Shared Model",
                "limit": {"context": 64_000, "output": 2_000},
            }
        },
    },
}


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def read(self) -> bytes:
        return self.payload


class ModelMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_path = Path(self.temporary_directory.name) / "models.json"
        self.data_path.write_text(json.dumps(FIXTURE), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_load_returns_all_raw_data_as_a_copy(self):
        first = load_model_data(self.data_path)
        self.assertEqual(first, FIXTURE)
        first["alpha"]["name"] = "Changed"
        self.assertEqual(load_model_data(self.data_path)["alpha"]["name"], "Alpha AI")

    def test_list_extracts_context_and_general_metadata(self):
        models = list_models(data_path=self.data_path)
        self.assertEqual(len(models), 4)
        unique = get_model_metadata("unique", data_path=self.data_path)
        self.assertEqual(unique["reference"], "alpha:unique")
        self.assertEqual(unique["context_window"], 32_000)
        self.assertEqual(unique["max_output_tokens"], 4_000)
        self.assertEqual(unique["description"], "A searchable description")
        self.assertEqual(unique["provider_metadata"]["doc"], "https://alpha.example")
        self.assertIn("alpha/unique", unique["references"])
        self.assertEqual(
            get_context_window("missing-context", data_path=self.data_path), 4_000
        )

    def test_qualified_and_provider_filtered_lookups(self):
        self.assertEqual(
            get_context_window("alpha:shared", data_path=self.data_path), 128_000
        )
        self.assertEqual(
            get_context_window("alpha/shared", data_path=self.data_path), 128_000
        )
        self.assertEqual(
            get_context_window("shared", provider="Beta AI", data_path=self.data_path),
            64_000,
        )

    def test_bare_duplicate_is_ambiguous(self):
        with self.assertRaises(AmbiguousModelError) as raised:
            get_model_metadata("shared", data_path=self.data_path)
        self.assertEqual(raised.exception.references, ["alpha:shared", "beta:shared"])

    def test_missing_model_context_uses_default(self):
        self.assertEqual(get_context_window("missing", data_path=self.data_path), 4_000)
        self.assertEqual(
            get_context_window("missing", default=8_192, data_path=self.data_path),
            8_192,
        )
        with self.assertRaises(ModelNotFoundError):
            get_model_metadata("missing", data_path=self.data_path)

    def test_provider_alias_and_google_prefix_are_supported(self):
        aliased = {
            "google-vertex": {
                "id": "google-vertex",
                "name": "Google Vertex",
                "models": {
                    "gemini": {
                        "id": "gemini",
                        "name": "Gemini",
                        "limit": {"context": 1_048_576, "output": 8_192},
                    }
                },
            }
        }
        self.data_path.write_text(json.dumps(aliased), encoding="utf-8")
        self.assertEqual(
            get_context_window(
                "models/gemini", provider="vertex", data_path=self.data_path
            ),
            1_048_576,
        )

    def test_query_searches_names_families_and_descriptions(self):
        self.assertEqual(
            [
                model["reference"]
                for model in query_models("searchable", data_path=self.data_path)
            ],
            ["alpha:unique"],
        )
        self.assertEqual(
            len(query_models("shared-family", data_path=self.data_path)), 1
        )
        self.assertEqual(
            len(query_models(provider="alpha", data_path=self.data_path)), 3
        )

    def test_refresh_validates_and_atomically_replaces_json(self):
        refreshed = {
            "new": {
                "id": "new",
                "name": "New",
                "models": {
                    "new-model": {
                        "id": "new-model",
                        "name": "New Model",
                        "limit": {"context": 1_000_000, "output": 10_000},
                    }
                },
            }
        }
        payload = json.dumps(refreshed).encode()
        with patch("llmai.models.urlopen", return_value=FakeResponse(payload)):
            result = refresh_model_data(destination=self.data_path)

        self.assertEqual(result, self.data_path.resolve())
        self.assertEqual(load_model_data(self.data_path), refreshed)
        self.assertEqual(
            get_context_window("new-model", data_path=self.data_path), 1_000_000
        )

    def test_refresh_keeps_existing_file_when_payload_is_invalid(self):
        original = self.data_path.read_bytes()
        with (
            patch(
                "llmai.models.urlopen", return_value=FakeResponse(b'{"invalid": true}')
            ),
            self.assertRaises(TypeError),
        ):
            refresh_model_data(destination=self.data_path)
        self.assertEqual(self.data_path.read_bytes(), original)


class BundledModelDataTests(unittest.TestCase):
    def test_bundled_snapshot_has_context_for_every_model(self):
        models = list_models()
        self.assertGreater(len(models), 5_000)
        self.assertTrue(
            all(
                isinstance(model["context_window"], int) and model["context_window"] > 0
                for model in models
            )
        )


if __name__ == "__main__":
    unittest.main()
