#!/usr/bin/env python3
"""Download open-access corpus sources into the local pdfs/ tree.

The manifest intentionally includes only sources with official or clearly open
PDF URLs. Paywalled books and pages that require manual export are documented
as skipped so the ingestion corpus does not pretend to contain sources that are
not actually present on disk.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "source_manifest.json"
REPORT_PATH = BASE_DIR / "source_download_report.json"

# A downloaded file must be at least this large to be plausibly a real source
# and not a short error/redirect stub. Size is a floor only -- content-type and
# magic-byte checks below are what actually decide validity.
MIN_VALID_BYTES = 10_000

# Expected format per output extension. Each entry lists the acceptable
# Content-Type prefixes and the leading magic-byte signatures the file may start
# with, so an HTML error/captcha page served in place of the real asset is
# rejected instead of accepted into the corpus.
_EXPECTED_FORMATS = {
    ".pdf": {
        "label": "PDF",
        "content_types": ("application/pdf", "application/x-pdf", "application/octet-stream"),
        "signatures": (b"%PDF-",),
    },
    ".epub": {
        "label": "EPUB",
        "content_types": ("application/epub+zip", "application/octet-stream"),
        "signatures": (b"PK\x03\x04",),
    },
}

# Signatures that betray an HTML page (error/captcha/"access denied") when we
# were expecting a binary document.
_HTML_SIGNATURES = (b"<!doctype html", b"<html", b"<head", b"<!--", b"<?xml")


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _looks_like_html(head: bytes) -> bool:
    stripped = head.lstrip().lower()
    return any(stripped.startswith(sig) for sig in _HTML_SIGNATURES)


def validate_download(output_path: Path, content_type: str) -> str | None:
    """Return a rejection reason if the file is not a valid expected source.

    Validation is deliberately not size-only: large HTML error/captcha pages
    easily clear a byte threshold. We confirm the declared Content-Type and the
    file's own magic bytes match the format implied by the output extension.
    Returns None when the file passes all checks.
    """
    size = output_path.stat().st_size
    if size < MIN_VALID_BYTES:
        return f"download too small ({size} bytes), likely an error page"

    expected = _EXPECTED_FORMATS.get(output_path.suffix.lower())
    with output_path.open("rb") as handle:
        head = handle.read(1024)

    # Unknown expected format: still refuse an obvious HTML page, but don't
    # enforce a specific signature we don't have.
    if expected is None:
        if _looks_like_html(head):
            return "response is an HTML page, not a downloadable document"
        return None

    label = expected["label"]
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if ctype and not any(ctype.startswith(ok) for ok in expected["content_types"]):
        return f"unexpected Content-Type '{ctype}' for {label} (likely an error/HTML page)"

    if not any(head.startswith(sig) for sig in expected["signatures"]):
        if _looks_like_html(head):
            return f"expected {label} but response body is an HTML page"
        return f"file header is not a valid {label} (magic bytes mismatch)"

    return None


def download(url: str, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Archipelago-source-downloader/1.0",
            "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
        },
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        content_type = response.headers.get("Content-Type", "")
        with output_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    reason = validate_download(output_path, content_type)
    if reason is not None:
        output_path.unlink(missing_ok=True)
        return {
            "status": "failed",
            "reason": reason,
        }

    return {
        "status": "downloaded",
        "http_status": status,
        "content_type": content_type,
        "bytes": output_path.stat().st_size,
    }


def main() -> int:
    results = []
    for item in load_manifest():
        title = item["title"]
        output = BASE_DIR / item["output_path"]
        url = item.get("url")

        if not url:
            results.append({
                **item,
                "status": "skipped",
                "reason": item.get("reason", "no open PDF URL in manifest"),
            })
            print(f"SKIP  {title}: {results[-1]['reason']}")
            continue

        if output.exists() and validate_download(output, "") is None:
            results.append({
                **item,
                "status": "exists",
                "bytes": output.stat().st_size,
            })
            print(f"KEEP  {title} -> {output.relative_to(BASE_DIR)}")
            continue

        try:
            result = download(url, output)
            results.append({**item, **result})
            if result["status"] == "downloaded":
                print(f"GET   {title} -> {output.relative_to(BASE_DIR)} ({result['bytes']} bytes)")
            else:
                print(f"FAIL  {title}: {result['reason']}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            output.unlink(missing_ok=True)
            results.append({**item, "status": "failed", "reason": str(exc)})
            print(f"FAIL  {title}: {exc}")

    REPORT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    downloaded = sum(1 for r in results if r["status"] in {"downloaded", "exists"})
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\nReport: {REPORT_PATH.relative_to(BASE_DIR)}")
    print(f"Local sources ready: {downloaded}; skipped: {skipped}; failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
