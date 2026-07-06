---
name: mineru-ocr
description: Use when Codex needs to parse PDFs, scanned documents, images, Office files, spreadsheets, presentations, or HTML with MinerU OCR / MinerU precise parsing API. Trigger for MinerU, OCR, document parsing, scanned PDF, table extraction, formula extraction, Markdown/JSON document extraction, and Chinese requests like "精准解析", "发给 MinerU", or "做 OCR".
---

# MinerU OCR

Use MinerU's precise parsing API by default. This is the token-based asynchronous API that returns a zip containing Markdown, JSON, and HTML outputs by default. Do not use the no-token Agent lightweight API unless the user explicitly asks for lightweight parsing.

For local GPU/offline parsing, `scripts/mineru_ocr.py` still exists, but it is not the default path for API-based work.

## Repository

Source repository: https://github.com/billerss11/mineru-ocr

Use this repository as the update source for this skill. The local git remote should be:

```powershell
git remote set-url origin https://github.com/billerss11/mineru-ocr.git
```

## Default Workflow

1. Confirm the file can be sent to an external API. Do not upload private or sensitive documents unless the user has approved that.
2. Ensure the API token is available as an environment variable. Prefer `MINERU_API_TOKEN`; never hard-code tokens into skill files, scripts, shell history, or output artifacts.
3. Run `scripts/mineru_api_ocr.py` with a local file path or an `http(s)` URL.
4. Poll until MinerU returns `done`.
5. Download and extract the result zip.
6. Return the paths to `full.md`, the HTML file, `*_content_list.json`, `*_middle.json` or `layout.json`, and `mineru_api_summary.json`.

## API Token Setup

Require a MinerU API token in the `MINERU_API_TOKEN` environment variable. Never put the token in `SKILL.md`, scripts, git commits, command examples, or chat messages.

One-time Windows user environment setup:

```powershell
[Environment]::SetEnvironmentVariable("MINERU_API_TOKEN", "paste-token-here", "User")
```

Restart Codex or the terminal after setting a user environment variable. For the current PowerShell session, load it without printing the token:

```powershell
$env:MINERU_API_TOKEN = [Environment]::GetEnvironmentVariable("MINERU_API_TOKEN", "User")
```

Check that it exists without revealing it:

```powershell
$token = [Environment]::GetEnvironmentVariable("MINERU_API_TOKEN", "User")
if ([string]::IsNullOrWhiteSpace($token)) { "MINERU_API_TOKEN is missing" } else { "MINERU_API_TOKEN is set, length=$($token.Length)" }
```

Temporary setup for only the current terminal:

```powershell
$env:MINERU_API_TOKEN = "paste-token-here"
```

Basic local file command:

```powershell
python "C:\Users\17999\.codex\skills\mineru-ocr\scripts\mineru_api_ocr.py" "C:\path\to\file.pdf" -o "C:\path\to\output"
```

URL command:

```powershell
python "C:\Users\17999\.codex\skills\mineru-ocr\scripts\mineru_api_ocr.py" "https://example.com/file.pdf" -o "C:\path\to\output"
```

Useful options:

```powershell
--lang en
--page-ranges "1-5,8"
--extra-format docx
--extra-format latex
--no-html
--no-ocr
--model-version pipeline
--model-version vlm
--model-version MinerU-HTML
```

## Precision Defaults

- Default model is `vlm` for PDFs, images, Office files, and other non-HTML sources.
- HTML source files always use `MinerU-HTML`, because MinerU requires that model for HTML input.
- OCR is enabled by default because scanned PDFs and images need it.
- Table recognition and formula recognition are enabled by default.
- HTML export is requested by default for non-HTML source files.
- Default language is `ch`. Use `--lang en` for English-only documents.
- For born-digital PDFs where OCR makes text worse, rerun with `--no-ocr`.
- If HTML output is not needed, rerun with `--no-html`.
- For academic, financial, or technical PDFs, inspect JSON outputs as well as `full.md`; Markdown alone may hide table or layout details.

## API Paths Used

- Local files: request upload URLs with `/api/v4/file-urls/batch`, upload by `PUT`, then poll `/api/v4/extract-results/batch/{batch_id}`.
- Public URLs: submit `/api/v4/extract/task`, then poll `/api/v4/extract/task/{task_id}`.
- Output zip URL is `full_zip_url`.

## Limits And Errors

MinerU precise API limits are 200 MB and 200 pages per file. Supported types include PDF, png, jpg/jpeg, jp2, webp, gif, bmp, doc/docx, ppt/pptx, xls/xlsx, and html.

Common failures:

- `A0202` or `A0211`: token is wrong or expired.
- `-60005`: file is over 200 MB.
- `-60006`: file has over 200 pages.
- `-60008`: MinerU cannot access the URL; upload the local file instead.
- `-60015` or `-60016`: conversion failed; try converting the file to PDF first.

Official docs: https://mineru.net/apiManage/docs
