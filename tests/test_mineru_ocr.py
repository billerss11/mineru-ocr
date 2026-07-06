import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mineru_ocr.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mineru_ocr", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class MinerUOcrTests(unittest.TestCase):
    def test_validate_model_root_accepts_pipeline_markers(self):
        mineru_ocr = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models" / "Layout" / "PP-DocLayoutV2").mkdir(parents=True)
            (root / "models" / "Layout" / "PP-DocLayoutV2" / "model.safetensors").write_bytes(b"x")
            (root / "models" / "OCR" / "paddleocr_torch").mkdir(parents=True)
            (root / "models" / "OCR" / "paddleocr_torch" / "ch_PP-OCRv6_small_det_infer.safetensors").write_bytes(b"x")
            (root / "models" / "OCR" / "paddleocr_torch" / "ch_PP-OCRv6_small_rec_infer.safetensors").write_bytes(b"x")
            (root / "models" / "TabRec" / "UnetStructure").mkdir(parents=True)
            (root / "models" / "TabRec" / "UnetStructure" / "unet.onnx").write_bytes(b"x")
            (root / "models" / "TabRec" / "SlanetPlus").mkdir(parents=True)
            (root / "models" / "TabRec" / "SlanetPlus" / "slanet-plus.onnx").write_bytes(b"x")
            (root / "models" / "TabCls" / "paddle_table_cls").mkdir(parents=True)
            (root / "models" / "TabCls" / "paddle_table_cls" / "PP-LCNet_x1_0_table_cls.onnx").write_bytes(b"x")

            mineru_ocr.validate_pipeline_model_root(root, table_enable=True)

    def test_validate_model_root_rejects_missing_markers(self):
        mineru_ocr = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(mineru_ocr.PreflightError, "missing required MinerU model files"):
                mineru_ocr.validate_pipeline_model_root(Path(tmp), table_enable=True)

    def test_require_gpu_fails_when_cuda_is_unavailable(self):
        mineru_ocr = load_module()
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: False,
                device_count=lambda: 0,
                get_device_name=lambda index: "unused",
            ),
            __version__="fake",
            version=SimpleNamespace(cuda=None),
        )

        with self.assertRaisesRegex(mineru_ocr.PreflightError, "GPU is required"):
            mineru_ocr.check_gpu(allow_cpu=False, torch_module=fake_torch)

    def test_write_summary_json_records_markdown_first_contract(self):
        mineru_ocr = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            markdown_path = output_dir / "doc" / "ocr" / "doc.md"
            content_list_path = output_dir / "doc" / "ocr" / "doc_content_list.json"
            middle_json_path = output_dir / "doc" / "ocr" / "doc_middle.json"
            markdown_path.parent.mkdir(parents=True)
            markdown_path.write_text("hello", encoding="utf-8")
            content_list_path.write_text("[]", encoding="utf-8")
            middle_json_path.write_text("{}", encoding="utf-8")

            summary_path = mineru_ocr.write_summary(
                output_dir=output_dir,
                input_path=Path("doc.pdf"),
                parse_dir=markdown_path.parent,
                markdown_path=markdown_path,
                content_list_path=content_list_path,
                middle_json_path=middle_json_path,
                backend="pipeline",
                method="ocr",
                gpu_info={"available": True, "device": "NVIDIA"},
                model_root=Path("C:/models"),
            )

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["primary_result"], str(markdown_path))
            self.assertEqual(summary["markdown_path"], str(markdown_path))
            self.assertEqual(summary["content_list_path"], str(content_list_path))
            self.assertEqual(summary["middle_json_path"], str(middle_json_path))
            self.assertEqual(summary["backend"], "pipeline")
            self.assertEqual(summary["method"], "ocr")


if __name__ == "__main__":
    unittest.main()
