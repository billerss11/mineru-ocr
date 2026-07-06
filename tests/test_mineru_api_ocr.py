import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mineru_api_ocr.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mineru_api_ocr", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    defaults = {
        "disable_formula": False,
        "disable_table": False,
        "lang": "ch",
        "extra_format": [],
        "no_html": False,
        "no_ocr": False,
        "data_id": None,
        "page_ranges": None,
        "no_cache": False,
        "cache_tolerance": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class MinerUApiOcrTests(unittest.TestCase):
    def test_resolve_model_version_uses_vlm_except_html(self):
        mineru = load_module()

        self.assertEqual(mineru.resolve_model_version("C:/docs/a.pdf", "auto"), "vlm")
        self.assertEqual(mineru.resolve_model_version("https://example.com/a.html", "auto"), "MinerU-HTML")
        self.assertEqual(mineru.resolve_model_version("https://example.com/a.html", "vlm"), "MinerU-HTML")
        self.assertEqual(mineru.resolve_model_version("C:/docs/a.pdf", "pipeline"), "pipeline")

    def test_cli_default_model_version_is_vlm(self):
        mineru = load_module()

        parsed = mineru.build_parser().parse_args(["demo.pdf"])

        self.assertEqual(parsed.model_version, "vlm")

    def test_url_payload_defaults_to_precise_ocr_options(self):
        mineru = load_module()

        payload = mineru.build_url_task_payload("https://example.com/a.pdf", args(), "vlm")

        self.assertEqual(payload["model_version"], "vlm")
        self.assertTrue(payload["is_ocr"])
        self.assertTrue(payload["enable_formula"])
        self.assertTrue(payload["enable_table"])
        self.assertEqual(payload["language"], "ch")
        self.assertEqual(payload["extra_formats"], ["html"])

    def test_no_html_disables_default_html_export(self):
        mineru = load_module()

        payload = mineru.build_url_task_payload("https://example.com/a.pdf", args(no_html=True), "vlm")

        self.assertNotIn("extra_formats", payload)

    def test_extra_formats_keep_default_html_and_user_formats(self):
        mineru = load_module()

        payload = mineru.build_url_task_payload("https://example.com/a.pdf", args(extra_format=["docx"]), "vlm")

        self.assertEqual(payload["extra_formats"], ["html", "docx"])

    def test_local_payload_puts_ocr_options_on_file_entry(self):
        mineru = load_module()

        payload = mineru.build_local_upload_payload(
            Path("demo.pdf"),
            args(data_id="abc-123", page_ranges="1-5", no_ocr=True),
            "vlm",
        )

        self.assertEqual(payload["model_version"], "vlm")
        self.assertEqual(payload["files"][0]["name"], "demo.pdf")
        self.assertEqual(payload["files"][0]["data_id"], "abc-123")
        self.assertEqual(payload["files"][0]["page_ranges"], "1-5")
        self.assertFalse(payload["files"][0]["is_ocr"])

    def test_read_token_from_env_removes_bearer_prefix(self):
        mineru = load_module()

        with patch.dict(os.environ, {"MINERU_API_TOKEN": "Bearer test-token"}):
            self.assertEqual(mineru.read_token_from_env("MINERU_API_TOKEN"), "test-token")

    def test_find_output_files_prefers_mineru_names(self):
        mineru = load_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "result"
            nested.mkdir()
            (nested / "full.md").write_text("# ok", encoding="utf-8")
            (nested / "full.html").write_text("<h1>ok</h1>", encoding="utf-8")
            (nested / "demo_content_list.json").write_text("[]", encoding="utf-8")
            (nested / "demo_middle.json").write_text("{}", encoding="utf-8")

            outputs = mineru.find_output_files(root)

            self.assertEqual(Path(outputs["markdown_path"]).name, "full.md")
            self.assertEqual(Path(outputs["html_path"]).name, "full.html")
            self.assertEqual(Path(outputs["content_list_path"]).name, "demo_content_list.json")
            self.assertEqual(Path(outputs["middle_json_path"]).name, "demo_middle.json")


if __name__ == "__main__":
    unittest.main()
