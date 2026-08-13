#!/usr/bin/env python
"""Submit files or URLs to the MinerU precise parsing API.

The script keeps API tokens out of command-line arguments by reading the token
from an environment variable. It uses the token-based precise API, not the
no-token lightweight Agent API.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

try:
    import requests as _requests
except Exception:  # pragma: no cover - optional dependency
    _requests = None


API_BASE = "https://mineru.net"
MAX_LOCAL_UPLOAD_BYTES = 200 * 1024 * 1024
SUPPORTED_SUFFIXES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".jp2",
    ".webp",
    ".gif",
    ".bmp",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".html",
}
IN_PROGRESS_STATES = {"waiting-file", "pending", "running", "converting"}
DONE_STATE = "done"
FAILED_STATE = "failed"


class MinerUApiError(RuntimeError):
    """Raised when MinerU API submission, polling, or download fails."""


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def input_suffix(value: str) -> str:
    if is_url(value):
        return Path(urllib.parse.urlparse(value).path).suffix.lower()
    return Path(value).suffix.lower()


def safe_stem(value: str, fallback: str = "mineru_result") -> str:
    if is_url(value):
        stem = Path(urllib.parse.urlparse(value).path).stem
    else:
        stem = Path(value).stem
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return (cleaned or fallback)[:80]


def read_token_from_env(env_name: str) -> str:
    token = os.environ.get(env_name, "").strip()
    if not token:
        raise MinerUApiError(
            f"Missing MinerU API token. Set ${env_name} first, for example: "
            f'$env:{env_name} = "your-token"'
        )
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        raise MinerUApiError(f"${env_name} is empty after removing the Bearer prefix.")
    return token


def parse_json_response(raw: str, url: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MinerUApiError(f"Non-JSON response from {url}: {raw[:500]}") from exc
    if not isinstance(payload, dict):
        raise MinerUApiError(f"Unexpected JSON response from {url}: {payload!r}")
    code = payload.get("code")
    if code not in (None, 0):
        raise MinerUApiError(f"MinerU API error {code}: {payload.get('msg', 'unknown error')}")
    return payload


def request_json(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    body = None
    headers = {"Accept": "*/*", "Authorization": f"Bearer {token}"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise MinerUApiError(f"HTTP {exc.code} from {url}: {raw[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise MinerUApiError(f"Network error calling {url}: {exc}") from exc
    return parse_json_response(raw, url)


def upload_file(upload_url: str, file_path: Path, timeout: float) -> None:
    if _requests is not None:
        with file_path.open("rb") as handle:
            response = _requests.put(
                upload_url,
                data=handle,
                headers={"Content-Type": ""},
                timeout=timeout,
            )
        if response.status_code not in {200, 201}:
            raise MinerUApiError(f"Upload failed with HTTP {response.status_code}: {response.text[:1000]}")
        return

    data = file_path.read_bytes()
    req = urllib.request.Request(
        upload_url,
        data=data,
        headers={"Accept": "*/*", "Content-Type": ""},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status not in {200, 201}:
                raise MinerUApiError(f"Upload failed with HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise MinerUApiError(f"Upload failed with HTTP {exc.code}: {raw[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise MinerUApiError(f"Network error during upload: {exc}") from exc


def download_file(url: str, output_path: Path, timeout: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _requests is not None:
        with _requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code != 200:
                raise MinerUApiError(f"Download failed with HTTP {response.status_code}: {response.text[:1000]}")
            with output_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        return

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    except urllib.error.URLError as exc:
        raise MinerUApiError(f"Network error downloading result zip: {exc}") from exc


def resolve_model_version(input_value: str, requested: str) -> str:
    if input_suffix(input_value) == ".html":
        return "MinerU-HTML"
    if requested == "auto":
        return "vlm"
    return requested


def build_common_payload(args: argparse.Namespace, model_version: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"model_version": model_version}
    if model_version != "MinerU-HTML":
        extra_formats = list(dict.fromkeys(args.extra_format or []))
        if not args.no_html and "html" not in extra_formats:
            extra_formats.insert(0, "html")
        payload.update(
            {
                "enable_formula": not args.disable_formula,
                "enable_table": not args.disable_table,
                "language": args.lang,
            }
        )
        if extra_formats:
            payload["extra_formats"] = extra_formats
    return payload


def build_url_task_payload(input_url: str, args: argparse.Namespace, model_version: str) -> dict[str, Any]:
    payload = build_common_payload(args, model_version)
    payload["url"] = input_url
    if model_version != "MinerU-HTML":
        payload["is_ocr"] = not args.no_ocr
    if args.data_id:
        payload["data_id"] = args.data_id
    if args.page_ranges:
        payload["page_ranges"] = args.page_ranges
    if args.no_cache:
        payload["no_cache"] = True
    if args.cache_tolerance is not None:
        payload["cache_tolerance"] = args.cache_tolerance
    return payload


def build_local_upload_payload(file_path: Path, args: argparse.Namespace, model_version: str) -> dict[str, Any]:
    payload = build_common_payload(args, model_version)
    file_entry: dict[str, Any] = {"name": file_path.name}
    if model_version != "MinerU-HTML":
        file_entry["is_ocr"] = not args.no_ocr
    if args.data_id:
        file_entry["data_id"] = args.data_id
    if args.page_ranges:
        file_entry["page_ranges"] = args.page_ranges
    payload["files"] = [file_entry]
    return payload


def validate_local_file(input_path: Path) -> None:
    if not input_path.exists():
        raise MinerUApiError(f"Input file does not exist: {input_path}")
    if not input_path.is_file():
        raise MinerUApiError(f"Input path is not a file: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise MinerUApiError(f"Unsupported file suffix '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
    size = input_path.stat().st_size
    if size > MAX_LOCAL_UPLOAD_BYTES:
        raise MinerUApiError(f"File is over 200 MB: {input_path} ({size} bytes)")


def submit_url_task(input_url: str, args: argparse.Namespace, token: str, model_version: str) -> dict[str, Any]:
    payload = build_url_task_payload(input_url, args, model_version)
    response = request_json(
        "POST",
        f"{args.api_base.rstrip('/')}/api/v4/extract/task",
        token,
        payload=payload,
        timeout=args.request_timeout,
    )
    task_id = response.get("data", {}).get("task_id")
    if not task_id:
        raise MinerUApiError(f"MinerU did not return task_id: {response}")
    return {"kind": "task", "id": task_id, "submit_response": response}


def submit_local_file(input_path: Path, args: argparse.Namespace, token: str, model_version: str) -> dict[str, Any]:
    validate_local_file(input_path)
    payload = build_local_upload_payload(input_path, args, model_version)
    response = request_json(
        "POST",
        f"{args.api_base.rstrip('/')}/api/v4/file-urls/batch",
        token,
        payload=payload,
        timeout=args.request_timeout,
    )
    data = response.get("data", {})
    batch_id = data.get("batch_id")
    upload_urls = data.get("file_urls") or []
    if not batch_id or not upload_urls:
        raise MinerUApiError(f"MinerU did not return batch_id and upload URL: {response}")
    upload_file(str(upload_urls[0]), input_path, timeout=args.request_timeout)
    return {"kind": "batch", "id": batch_id, "submit_response": response}


def log_progress(args: argparse.Namespace, message: str) -> None:
    if args.verbose:
        print(message, file=sys.stderr)


def poll_task(task_id: str, args: argparse.Namespace, token: str) -> dict[str, Any]:
    deadline = time.time() + args.timeout
    url = f"{args.api_base.rstrip('/')}/api/v4/extract/task/{task_id}"
    while True:
        response = request_json("GET", url, token, timeout=args.request_timeout)
        data = response.get("data", {})
        state = data.get("state")
        log_progress(args, f"task {task_id}: {state}")
        if state == DONE_STATE:
            return data
        if state == FAILED_STATE:
            raise MinerUApiError(f"MinerU task failed: {data.get('err_msg', '')}")
        if state not in IN_PROGRESS_STATES:
            raise MinerUApiError(f"Unknown MinerU task state '{state}': {data}")
        if time.time() >= deadline:
            raise MinerUApiError(f"Timed out waiting for task {task_id}")
        time.sleep(args.poll_interval)


def poll_batch(batch_id: str, args: argparse.Namespace, token: str) -> dict[str, Any]:
    deadline = time.time() + args.timeout
    url = f"{args.api_base.rstrip('/')}/api/v4/extract-results/batch/{batch_id}"
    while True:
        response = request_json("GET", url, token, timeout=args.request_timeout)
        data = response.get("data", {})
        results = data.get("extract_result") or []
        states = [item.get("state") for item in results]
        log_progress(args, f"batch {batch_id}: {states or ['waiting']}")
        failed = [item for item in results if item.get("state") == FAILED_STATE]
        if failed:
            raise MinerUApiError(f"MinerU batch failed: {failed[0].get('err_msg', '')}")
        if results and all(item.get("state") == DONE_STATE for item in results):
            return data
        unknown = [state for state in states if state not in IN_PROGRESS_STATES and state != DONE_STATE]
        if unknown:
            raise MinerUApiError(f"Unknown MinerU batch state {unknown}: {data}")
        if time.time() >= deadline:
            raise MinerUApiError(f"Timed out waiting for batch {batch_id}")
        time.sleep(args.poll_interval)


def result_zip_url(job: dict[str, Any], final_data: dict[str, Any]) -> str:
    if job["kind"] == "task":
        zip_url = final_data.get("full_zip_url")
    else:
        results = final_data.get("extract_result") or []
        zip_url = results[0].get("full_zip_url") if results else None
    if not zip_url:
        raise MinerUApiError(f"MinerU result did not include full_zip_url: {final_data}")
    return str(zip_url)


def safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    base = extract_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (extract_dir / member.filename).resolve()
            try:
                target.relative_to(base)
            except ValueError as exc:
                raise MinerUApiError(f"Unsafe path in zip: {member.filename}") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def first_path(paths: list[Path]) -> str | None:
    return str(paths[0].resolve()) if paths else None


def find_output_files(extract_dir: Path) -> dict[str, Any]:
    markdown = sorted(extract_dir.rglob("full.md")) or sorted(extract_dir.rglob("*.md"))
    html = sorted(extract_dir.rglob("*.html"))
    content_list = sorted(extract_dir.rglob("*_content_list.json"))
    middle = sorted(extract_dir.rglob("*_middle.json")) or sorted(extract_dir.rglob("layout.json"))
    model = sorted(extract_dir.rglob("*_model.json"))
    json_files = sorted(extract_dir.rglob("*.json"))
    return {
        "markdown_path": first_path(markdown),
        "html_path": first_path(html),
        "html_paths": [str(path.resolve()) for path in html],
        "content_list_path": first_path(content_list),
        "middle_json_path": first_path(middle),
        "model_json_path": first_path(model),
        "json_paths": [str(path.resolve()) for path in json_files],
    }


def write_summary(output_dir: Path, name: str, summary: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{name}_mineru_api_summary.json"
    summary["summary_path"] = str(summary_path.resolve())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    token = read_token_from_env(args.token_env)
    model_version = resolve_model_version(args.input, args.model_version)
    output_dir = Path(args.output_dir).expanduser().resolve()
    name = safe_stem(args.data_id or args.input)

    if is_url(args.input):
        job = submit_url_task(args.input, args, token, model_version)
        final_data = poll_task(job["id"], args, token)
    else:
        input_path = Path(args.input).expanduser().resolve()
        job = submit_local_file(input_path, args, token, model_version)
        final_data = poll_batch(job["id"], args, token)

    zip_url = result_zip_url(job, final_data)
    summary: dict[str, Any] = {
        "input": args.input,
        "api_base": args.api_base.rstrip("/"),
        "job_kind": job["kind"],
        "job_id": job["id"],
        "model_version": model_version,
        "full_zip_url": zip_url,
        "final_data": final_data,
    }

    if not args.no_download:
        zip_path = output_dir / f"{name}_mineru_result.zip"
        extract_dir = output_dir / f"{name}_mineru_result"
        download_file(zip_url, zip_path, timeout=args.request_timeout)
        safe_extract_zip(zip_path, extract_dir)
        summary.update(
            {
                "zip_path": str(zip_path.resolve()),
                "extract_dir": str(extract_dir.resolve()),
                **find_output_files(extract_dir),
            }
        )

    summary_path = write_summary(output_dir, name, summary)
    summary["summary_path"] = str(summary_path.resolve())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MinerU precise API OCR for a local file or public URL.")
    parser.add_argument("input", help="Local file path or http(s) URL.")
    parser.add_argument("-o", "--output-dir", default="mineru_api_output", help="Output directory.")
    parser.add_argument("--token-env", default="MINERU_API_TOKEN", help="Environment variable containing the API token.")
    parser.add_argument(
        "--model-version",
        choices=["auto", "pipeline", "vlm", "MinerU-HTML"],
        default="vlm",
        help="MinerU model version. Default: vlm. HTML inputs always use MinerU-HTML.",
    )
    parser.add_argument("--lang", default="ch", help="Document language hint for pipeline/vlm. Default: ch.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR for born-digital PDFs.")
    parser.add_argument("--disable-table", action="store_true", help="Disable table recognition.")
    parser.add_argument("--disable-formula", action="store_true", help="Disable formula recognition.")
    parser.add_argument("--extra-format", action="append", choices=["docx", "html", "latex"], default=[], help="Extra export format. Repeatable. HTML is requested by default.")
    parser.add_argument("--no-html", action="store_true", help="Do not request the default HTML export.")
    parser.add_argument("--page-ranges", default=None, help='Page range, for example "1-5,8" or "2--2".')
    parser.add_argument("--data-id", default=None, help="Optional business data ID sent to MinerU.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass URL cache for URL tasks.")
    parser.add_argument("--cache-tolerance", type=int, default=None, help="URL cache tolerance in seconds.")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval in seconds.")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Overall polling timeout in seconds.")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout in seconds.")
    parser.add_argument("--api-base", default=API_BASE, help="MinerU API base URL.")
    parser.add_argument("--no-download", action="store_true", help="Only return full_zip_url; do not download or extract it.")
    parser.add_argument("--verbose", action="store_true", help="Print polling progress to stderr.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except MinerUApiError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
