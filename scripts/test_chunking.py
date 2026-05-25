#!/usr/bin/env python3
"""
Diagnostic script — run parse + chunk on a PDF and print detailed output.
Shows every raw element and every resulting chunk so you can spot bad splits.

Usage:
    uv run scripts/test_chunking.py data/uploads/docling_report-3-5.pdf
    uv run scripts/test_chunking.py data/uploads/docling_report-3-5.pdf --raw   # raw elements only
    uv run scripts/test_chunking.py data/uploads/docling_report-3-5.pdf --chunks # chunks only
"""

import sys
import argparse
from pathlib import Path

# Add src to path so imports work without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, Header, Table, ListItem

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

TYPE_COLOR = {
    "title": BOLD + CYAN,
    "table": YELLOW,
    "list":  GREEN,
    "text":  RESET,
}


def _classify(el) -> str:
    if isinstance(el, (Title, Header)):
        return "title"
    if isinstance(el, Table):
        return "table"
    if isinstance(el, ListItem):
        return "list"
    return "text"


def _page(el) -> int:
    try:
        return el.metadata.page_number or 1
    except AttributeError:
        return 1


def print_raw_elements(pdf_path: Path) -> list:
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}RAW ELEMENTS  —  {pdf_path.name}{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")

    raw = partition_pdf(filename=str(pdf_path), strategy="fast", include_page_breaks=False)

    type_counts: dict[str, int] = {}
    for i, el in enumerate(raw):
        text = el.text.strip() if el.text else ""
        etype = _classify(el)
        page = _page(el)
        type_counts[etype] = type_counts.get(etype, 0) + 1
        color = TYPE_COLOR[etype]
        marker = "⚠  " if len(text) < 5 and text else "   "

        print(f"{DIM}[{i:04d}]{RESET} {color}{etype:6}{RESET}  p{page:03d}  {marker}{repr(text[:120])}")

    print(f"\n{BOLD}Summary:{RESET}")
    for k, v in sorted(type_counts.items()):
        print(f"  {k:8}: {v}")
    print(f"  {'total':8}: {len(raw)}")
    return raw


def print_chunks(pdf_path: Path) -> None:
    import tiktoken
    import hashlib

    enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(enc.encode(text))

    def overlap_prefix(text: str, n: int) -> str:
        tokens = enc.encode(text)
        return enc.decode(tokens[-n:]) if len(tokens) > n else text

    raw = partition_pdf(filename=str(pdf_path), strategy="fast", include_page_breaks=False)

    MAX_TOKENS = 512
    OVERLAP = 50

    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}CHUNKS  —  {pdf_path.name}  (max={MAX_TOKENS} tok, overlap={OVERLAP}){RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")

    chunks = []
    section = "Introduction"
    texts: list[str] = []
    pages: list[int] = []
    tokens = 0

    def flush():
        nonlocal texts, pages, tokens
        if not texts:
            return
        text = "\n".join(texts)
        chunks.append({
            "index": len(chunks),
            "section": section,
            "text": text,
            "tokens": tokens,
            "page_start": min(pages),
            "page_end": max(pages),
        })
        ov = overlap_prefix(text, OVERLAP)
        texts = [ov] if ov else []
        pages = [pages[-1]]
        tokens = count_tokens(ov)

    for el in raw:
        text = el.text.strip() if el.text else ""
        if not text:
            continue
        etype = _classify(el)
        page = _page(el)

        if etype == "title":
            flush()
            section = text
            continue

        el_tok = count_tokens(text)
        if tokens + el_tok > MAX_TOKENS and texts:
            flush()

        texts.append(text)
        pages.append(page)
        tokens += el_tok

    flush()

    short_chunks = 0
    for c in chunks:
        is_short = c["tokens"] < 20
        if is_short:
            short_chunks += 1
        color = RED if is_short else RESET
        print(
            f"{color}{BOLD}[Chunk {c['index']:03d}]{RESET}  "
            f"tok={c['tokens']:4d}  p{c['page_start']}-{c['page_end']}  "
            f"section={repr(c['section'][:50])}"
        )
        print(f"  {DIM}{repr(c['text'][:200])}{RESET}")
        print()

    print(f"{BOLD}Total chunks : {len(chunks)}{RESET}")
    print(f"{RED}Short (<20 tok): {short_chunks}{RESET}" if short_chunks else f"{GREEN}Short (<20 tok): 0{RESET}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose PDF chunking")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument("--raw", action="store_true", help="Show raw elements only")
    parser.add_argument("--chunks", action="store_true", help="Show chunks only")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"{RED}File not found: {args.pdf}{RESET}")
        sys.exit(1)

    show_raw = args.raw or (not args.raw and not args.chunks)
    show_chunks = args.chunks or (not args.raw and not args.chunks)

    if show_raw:
        print_raw_elements(args.pdf)
    if show_chunks:
        print_chunks(args.pdf)


if __name__ == "__main__":
    main()
