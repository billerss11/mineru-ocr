#!/usr/bin/env python
"""Offline MinerU OCR wrapper for Codex skills.

The script intentionally performs preflight checks before importing MinerU.
If local model files or GPU are missing, it exits before MinerU can download
models implicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


class PreflightError(RuntimeError):
    """Raised when OCR should not start because a required local resource is missing."""


DEFAULT_MODEL_ROOTS = [
    Path(os.environ.get("MINERU_PIPELINE_MODEL_ROOT", "")),
    Path.home() / ".cache" / "modelscope" / "hub" / "models" / "OpenDataLab" / "PDF-Extract-Kit-1___0",
    Path.home() / ".cache" / "modelscope" / "hub" / "models" / "OpenDataLab" / "PDF-Extract-Kit-1.0",
    Path.home() / ".cache" / "huggingface" / "hub" / "models--opendatalab--PDF-Extract-Kit-1.0" / "snapshots",
]


def _existing_hf_snapshot(root: Path) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    snapshots = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0] if snapshots else None


def find_model_root(explicit_model_root: str | None = None) -> Path:
    candidates = [Path(explicit_model_root)] if explicit_model_root else DEFAULT_MODEL_ROOTS
    for candidate in candidates:
        if not str(candidate):
            continue
        if candidate.name == "snapshots":
            snapshot = _existing_hf_snapshot(candidate)
            if snapshot is not None:
                return snapshot.resolve()
        if candidate.exists():
            return candidate.resolve()
    raise PreflightError(
        "No local MinerU pipeline model root found. Pass --model-root or set MINERU_PIPELINE_MODEL_ROOT. "
        "Stopped before invoking MinerU to avoid downloading models."
    )


def _required_pipeline_markers(table_enable: bool, formula_enable: bool) -> list[Path]:
    markers = [
        Path("models/Layout/PP-DocLayoutV2/model.safetensors"),
        Path("models/OCR/paddleocr_torch/ch_PP-OCRv6_small_det_infer.safetensors"),
        Path("models/OCR/paddleocr_torch/ch_PP-OCRv6_small_rec_infer.safetensors"),
    ]
    if table_enable:
        markers.extend(
            [
                Path("models/TabRec/UnetStructure/unet.onnx"),
                Path("models/TabRec/SlanetPlus/slanet-plus.onnx"),
                Path("models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx"),
            ]
        )
    if formula_enable:
        markers.extend(
            [
                Path("models/MFR/pp_formulanet_plus_m"),
            ]
        )
    return markers


def validate_pipeline_model_root(model_root: Path, table_enable: bool, formula_enable: bool = False) -> None:
    missing = [marker for marker in _required_pipeline_markers(table_enable, formula_enable) if not (model_root / marker).exists()]
    if missing:
        details = "\n".join(f"- {model_root / marker}" for marker in missing)
        raise PreflightError(
            "Local model root is missing required MinerU model files. "
            "Stopped before invoking MinerU to avoid downloading models.\n"
            f"{details}"
        )


def check_gpu(allow_cpu: bool, torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as exc:
            raise PreflightError(f"Cannot import torch: {exc}") from exc

    available = bool(torch_module.cuda.is_available())
    info = {
        "available": available,
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "device_count": int(torch_module.cuda.device_count()) if hasattr(torch_module.cuda, "device_count") else 0,
        "device": torch_module.cuda.get_device_name(0) if available else None,
    }
    if not available and not allow_cpu:
        raise PreflightError(
            "GPU is required but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled torch build in the OCR environment, or rerun with --allow-cpu."
        )
    return info


def write_mineru_local_config(config_path: Path, model_root: Path) -> None:
    payload = {
        "models-dir": {
            "pipeline": str(model_root),
            "vlm": "",
        },
        "config_version": "1.3.1",
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary(
    output_dir: Path,
    input_path: Path,
    parse_dir: Path,
    markdown_path: Path,
    content_list_path: Path,
    middle_json_path: Path,
    backend: str,
    method: str,
    gpu_info: dict[str, Any],
    model_root: Path,
) -> Path:
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "parse_dir": str(parse_dir),
        "primary_result": str(markdown_path),
        "markdown_path": str(markdown_path),
        "content_list_path": str(content_list_path),
        "middle_json_path": str(middle_json_path),
        "backend": backend,
        "method": method,
        "gpu": gpu_info,
        "model_root": str(model_root),
    }
    summary_path = parse_dir / "mineru_ocr_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MinerU OCR on a local PDF or image without implicit model downloads.")
    parser.add_argument("input_path", help="Local PDF or image path.")
    parser.add_argument("-o", "--output-dir", default="mineru_ocr_output", help="Output directory. Default: mineru_ocr_output")
    parser.add_argument("--model-root", default=None, help="Local PDF-Extract-Kit-1.0 model root containing the models/ directory.")
    parser.add_argument("--method", choices=["auto", "txt", "ocr"], default="ocr", help="MinerU parse method. Default: ocr")
    parser.add_argument("--lang", default="en", help="OCR language hint. Default: en")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback when CUDA is unavailable.")
    parser.add_argument("--no-table", action="store_true", help="Disable table recognition.")
    parser.add_argument("--formula", action="store_true", help="Enable formula recognition. Requires local formula models.")
    parser.add_argument("--start-page", type=int, default=0, help="Zero-based start page for PDFs.")
    parser.add_argument("--end-page", type=int, default=None, help="Zero-based end page for PDFs.")
    parser.add_argument("--env-check", action="store_true", help="Run preflight checks only; do not invoke MinerU.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise PreflightError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise PreflightError(f"Input path must be a file: {input_path}")

    table_enable = not args.no_table
    model_root = find_model_root(args.model_root)
    validate_pipeline_model_root(model_root, table_enable=table_enable, formula_enable=args.formula)
    gpu_info = check_gpu(allow_cpu=args.allow_cpu)

    output_dir = Path(args.output_dir).expanduser().resolve()
    parse_dir = output_dir / input_path.stem / args.method
    markdown_path = parse_dir / f"{input_path.stem}.md"
    content_list_path = parse_dir / f"{input_path.stem}_content_list.json"
    middle_json_path = parse_dir / f"{input_path.stem}_middle.json"

    if args.env_check:
        return {
            "input_path": str(input_path),
            "model_root": str(model_root),
            "gpu": gpu_info,
            "would_write": str(parse_dir),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mineru_ocr_") as tmp:
        config_path = Path(tmp) / "mineru.json"
        write_mineru_local_config(config_path, model_root)
        os.environ["MINERU_MODEL_SOURCE"] = "local"
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(config_path)
        if gpu_info["available"]:
            os.environ["MINERU_DEVICE_MODE"] = "cuda"
        elif args.allow_cpu:
            os.environ["MINERU_DEVICE_MODE"] = "cpu"

        from mineru.cli.common import do_parse, read_fn
        from mineru.utils.enum_class import MakeMode

        do_parse(
            output_dir=str(output_dir),
            pdf_file_names=[input_path.stem],
            pdf_bytes_list=[read_fn(input_path)],
            p_lang_list=[args.lang],
            backend="pipeline",
            parse_method=args.method,
            formula_enable=args.formula,
            table_enable=table_enable,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=True,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
            f_make_md_mode=MakeMode.MM_MD,
            start_page_id=args.start_page,
            end_page_id=args.end_page,
        )

    if not markdown_path.exists():
        raise PreflightError(f"MinerU finished but markdown output was not found: {markdown_path}")

    summary_path = write_summary(
        output_dir=output_dir,
        input_path=input_path,
        parse_dir=parse_dir,
        markdown_path=markdown_path,
        content_list_path=content_list_path,
        middle_json_path=middle_json_path,
        backend="pipeline",
        method=args.method,
        gpu_info=gpu_info,
        model_root=model_root,
    )
    return {
        "summary_path": str(summary_path),
        "markdown_path": str(markdown_path),
        "content_list_path": str(content_list_path),
        "middle_json_path": str(middle_json_path),
        "gpu": gpu_info,
        "model_root": str(model_root),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except PreflightError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
