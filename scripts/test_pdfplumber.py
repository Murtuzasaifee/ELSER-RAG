#!/usr/bin/env python3
"""
Test pdfplumber-based parsing approach — generic, font/size agnostic.

Heading detection uses two signals that work across ALL PDFs:
  1. Font weight: fontname contains bold/medium weight keywords
  2. Relative size: font size > median body size * 1.15 on that page

Word spacing is reconstructed from character bounding boxes using a gap
threshold relative to the font size (adaptive, not hardcoded pixels).

Usage:
    uv run scripts/test_pdfplumber.py data/uploads/docling_report-3-5.pdf
    uv run scripts/test_pdfplumber.py data/uploads/docling_report-3-5.pdf --elements
    uv run scripts/test_pdfplumber.py data/uploads/docling_report-3-5.pdf --chunks
    uv run scripts/test_pdfplumber.py data/uploads/docling_report-3-5.pdf --compare
    uv run scripts/test_pdfplumber.py data/uploads/docling_report-3-5.pdf --fonts    # show all fonts found
"""

import sys
import argparse
import statistics
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import tiktoken

RESET   = "\033[0m"
BOLD    = "\033[1m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
DIM     = "\033[2m"

_enc = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
# Keywords in fontname that signal bold/heading weight — covers 99% of fonts
_BOLD_KEYWORDS = {"bold", "medi", "heavy", "black", "semibold", "demi", "extrab"}

# Heading must have more chars than this — kills single-letter artefacts
MIN_HEADING_CHARS = 4

# Relative size multiplier: line font_size > median_body * this = heading
RELATIVE_SIZE_MULTIPLIER = 1.15

# Space insertion: gap between adjacent chars > font_size * this = insert space
SPACE_GAP_RATIO = 0.20

# Page footer/header zones — skip text here (page numbers, running headers)
FOOTER_FRACTION = 0.07
HEADER_FRACTION = 0.05

# Max tokens per chunk, overlap
MAX_CHUNK_TOKENS = 512
OVERLAP_TOKENS   = 50
# ---------------------------------------------------------------------------


@dataclass
class Element:
    kind: str   # "title" | "text" | "table"
    text: str
    page: int


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _overlap_prefix(text: str, n: int) -> str:
    tokens = _enc.encode(text)
    return _enc.decode(tokens[-n:]) if len(tokens) > n else text


# ---------------------------------------------------------------------------
# Generic heading detection
# ---------------------------------------------------------------------------

def _is_bold_font(fontname: str) -> bool:
    """Check font weight keywords — works across all font families."""
    name = fontname.lower()
    return any(kw in name for kw in _BOLD_KEYWORDS)


def _median_body_size(chars: list[dict]) -> float:
    """Median font size of non-bold chars on the page = body text size."""
    sizes = [
        c["size"] for c in chars
        if c.get("size") and not _is_bold_font(c.get("fontname", ""))
    ]
    if not sizes:
        return 10.0
    return statistics.median(sizes)


def _line_is_heading(line_chars: list[dict], median_body: float) -> bool:
    """
    A line is a heading if:
      - Majority of its chars use a bold/medium weight font, OR
      - Its dominant font size is >= median_body * RELATIVE_SIZE_MULTIPLIER
    """
    if not line_chars:
        return False
    bold_count = sum(1 for c in line_chars if _is_bold_font(c.get("fontname", "")))
    bold_ratio = bold_count / len(line_chars)
    if bold_ratio >= 0.6:
        return True
    sizes = [c["size"] for c in line_chars if c.get("size")]
    if sizes and max(sizes) >= median_body * RELATIVE_SIZE_MULTIPLIER:
        return True
    return False


# ---------------------------------------------------------------------------
# Character-level text reconstruction with adaptive spacing
# ---------------------------------------------------------------------------

def _chars_to_text(chars: list[dict]) -> str:
    """
    Reconstruct text from sorted char list, inserting spaces where the gap
    between adjacent characters exceeds SPACE_GAP_RATIO * font_size.
    This fixes word-joining in PDFs with unusual font encodings.
    """
    if not chars:
        return ""
    # Sort left to right
    chars = sorted(chars, key=lambda c: c["x0"])
    parts = []
    for i, c in enumerate(chars):
        ch = c.get("text", "")
        if not ch or ch == " ":
            if ch == " ":
                parts.append(" ")
            continue
        parts.append(ch)
        if i < len(chars) - 1:
            gap = chars[i + 1]["x0"] - c["x1"]
            font_size = c.get("size", 10) or 10
            if gap > font_size * SPACE_GAP_RATIO:
                parts.append(" ")
    return "".join(parts).strip()


# ---------------------------------------------------------------------------
# Table markdown
# ---------------------------------------------------------------------------

def _table_to_markdown(rows: list[list]) -> str:
    if not rows:
        return ""
    cleaned = [
        [str(cell).replace("\n", " ").strip() if cell is not None else "" for cell in row]
        for row in rows
    ]
    # Pad all rows to same width
    width = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    header = cleaned[0]
    body   = cleaned[1:]
    lines  = ["| " + " | ".join(header) + " |",
              "| " + " | ".join(["---"] * width) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_pdf_pdfplumber(pdf_path: Path) -> list[Element]:
    elements: list[Element] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_num    = page.page_number
            page_height = float(page.height)
            all_chars   = page.chars

            median_body = _median_body_size(all_chars)

            # ---- 1. Detect and extract tables (bordered tables) ----
            table_bboxes: list[tuple] = []
            for table in page.find_tables():
                rows = table.extract()
                if not rows:
                    continue
                md = _table_to_markdown(rows)
                if md:
                    elements.append(Element(kind="table", text=md, page=page_num))
                    table_bboxes.append(table.bbox)

            def _in_table(x0, y0, x1, y1) -> bool:
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    if x0 < tx1 and x1 > tx0 and y0 < ty1 and y1 > ty0:
                        return True
                return False

            # ---- 2. Filter chars outside tables and outside header/footer ----
            body_chars = [
                c for c in all_chars
                if c.get("text", "").strip()
                and not _in_table(c["x0"], c["top"], c["x1"], c["bottom"])
                and c["top"] > page_height * HEADER_FRACTION
                and c["bottom"] < page_height * (1 - FOOTER_FRACTION)
            ]

            # ---- 3. Group chars into lines by vertical position (top, 2px snap) ----
            lines: dict[int, list[dict]] = {}
            for c in body_chars:
                bucket = round(c["top"] / 2) * 2  # snap to 2px grid
                lines.setdefault(bucket, []).append(c)

            # ---- 4. Process each line top-to-bottom ----
            for bucket in sorted(lines):
                line_chars = sorted(lines[bucket], key=lambda c: c["x0"])
                text = _chars_to_text(line_chars)
                if not text or len(text.strip()) < 2:
                    continue

                is_heading = (
                    _line_is_heading(line_chars, median_body)
                    and len(text.strip()) >= MIN_HEADING_CHARS
                )
                elements.append(Element(
                    kind="title" if is_heading else "text",
                    text=text.strip(),
                    page=page_num,
                ))

    return elements


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_elements(elements: list[Element]) -> list[dict]:
    chunks: list[dict] = []
    section = "Introduction"
    texts: list[str] = []
    pages:  list[int] = []
    tokens = 0

    def flush():
        nonlocal texts, pages, tokens
        if not texts:
            return
        text = "\n".join(texts)
        chunks.append({
            "index":      len(chunks),
            "section":    section,
            "text":       text,
            "tokens":     tokens,
            "page_start": min(pages),
            "page_end":   max(pages),
        })
        ov    = _overlap_prefix(text, OVERLAP_TOKENS)
        texts  = [ov] if ov else []
        pages  = [pages[-1]]
        tokens = _count_tokens(ov)

    for el in elements:
        if el.kind == "title":
            flush()
            section = el.text
            continue

        el_tok = _count_tokens(el.text)

        if el.kind == "table":
            if texts:
                flush()
            texts.append(el.text)
            pages.append(el.page)
            tokens += el_tok
            flush()
            continue

        if tokens + el_tok > MAX_CHUNK_TOKENS and texts:
            flush()

        texts.append(el.text)
        pages.append(el.page)
        tokens += el_tok

    flush()
    return chunks


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_fonts(pdf_path: Path) -> None:
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}ALL FONTS FOUND IN PDF{RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")
    seen: dict[tuple, list[str]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for c in page.chars:
                key = (round(c.get("size", 0), 1), c.get("fontname", "?").split("+")[-1])
                seen.setdefault(key, []).append(c.get("text", ""))
    for (size, font), samples in sorted(seen.items()):
        bold_flag = f"{CYAN}[BOLD]{RESET}" if _is_bold_font(font) else "      "
        sample    = "".join(samples[:20]).replace("\n", "")[:40]
        print(f"  size={size:5.1f}  {bold_flag}  font={font:40s}  sample={repr(sample)}")


def print_elements(elements: list[Element]) -> None:
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}PDFPLUMBER ELEMENTS  (generic heading detection){RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")
    counts: dict[str, int] = {}
    for i, el in enumerate(elements):
        counts[el.kind] = counts.get(el.kind, 0) + 1
        color   = CYAN if el.kind == "title" else (YELLOW if el.kind == "table" else RESET)
        flag    = "⚠  " if len(el.text) < 5 else "   "
        preview = repr(el.text[:120]) if el.kind != "table" else f"[TABLE rows={el.text.count(chr(10))+1}]"
        print(f"{DIM}[{i:04d}]{RESET} {color}{el.kind:6}{RESET}  p{el.page:03d}  {flag}{preview}")
    print(f"\n{BOLD}Summary:{RESET}")
    for k, v in sorted(counts.items()):
        print(f"  {k:8}: {v}")
    print(f"  {'total':8}: {len(elements)}")


def print_chunks(chunks: list[dict]) -> None:
    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}PDFPLUMBER CHUNKS  (max={MAX_CHUNK_TOKENS} tok, overlap={OVERLAP_TOKENS}){RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")
    short = 0
    for c in chunks:
        is_short = c["tokens"] < 20
        if is_short:
            short += 1
        color = RED if is_short else RESET
        print(
            f"{color}{BOLD}[Chunk {c['index']:03d}]{RESET}  "
            f"tok={c['tokens']:4d}  p{c['page_start']}-{c['page_end']}  "
            f"section={repr(c['section'][:60])}"
        )
        print(f"  {DIM}{repr(c['text'][:250])}{RESET}")
        print()
    print(f"{BOLD}Total chunks: {len(chunks)}{RESET}")
    if short:
        print(f"{RED}Short (<20 tok): {short}{RESET}")
    else:
        print(f"{GREEN}Short (<20 tok): 0  ✓{RESET}")


def print_comparison(pdf_path: Path, plumber_chunks: list[dict]) -> None:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.documents.elements import Title, Header, Table, ListItem
    import tiktoken as t
    enc = t.get_encoding("cl100k_base")

    def tok(text): return len(enc.encode(text))
    def classify(el):
        if isinstance(el, (Title, Header)): return "title"
        if isinstance(el, Table):           return "table"
        if isinstance(el, ListItem):        return "list"
        return "text"

    raw = partition_pdf(filename=str(pdf_path), strategy="fast", include_page_breaks=False)
    uchunks, section = [], "Introduction"
    texts, pages, tokens = [], [], 0

    def uflush():
        nonlocal texts, pages, tokens
        if not texts: return
        text = "\n".join(texts)
        uchunks.append({"tokens": tok(text), "text": text})
        t_enc = enc.encode(text)
        ov = enc.decode(t_enc[-50:]) if len(t_enc) > 50 else text
        texts, pages, tokens = [ov] if ov else [], [pages[-1]] if pages else [], tok(ov)

    for el in raw:
        text = (el.text or "").strip()
        if not text: continue
        if classify(el) == "title": uflush(); section = text; continue
        et = tok(text)
        if tokens + et > 512 and texts: uflush()
        texts.append(text); pages.append(1); tokens += et
    uflush()

    u_short = sum(1 for c in uchunks if c["tokens"] < 20)
    p_short = sum(1 for c in plumber_chunks if c["tokens"] < 20)
    u_avg   = sum(c["tokens"] for c in uchunks) / len(uchunks) if uchunks else 0
    p_avg   = sum(c["tokens"] for c in plumber_chunks) / len(plumber_chunks) if plumber_chunks else 0

    print(f"\n{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}COMPARISON: unstructured fast  vs  pdfplumber (generic){RESET}")
    print(f"{BOLD}{'='*80}{RESET}\n")
    print(f"{'Metric':<35} {'unstructured':>18} {'pdfplumber':>18}")
    print("-" * 73)
    print(f"{'Total chunks':<35} {len(uchunks):>18} {len(plumber_chunks):>18}")
    sc = RED if u_short > p_short else RESET
    print(f"{'Short chunks (<20 tok)':<35} {sc}{u_short:>18}{RESET} {GREEN}{p_short:>18}{RESET}")
    print(f"{'Avg tokens / chunk':<35} {u_avg:>18.1f} {p_avg:>18.1f}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test generic pdfplumber chunking")
    parser.add_argument("pdf",        type=Path)
    parser.add_argument("--elements", action="store_true")
    parser.add_argument("--chunks",   action="store_true")
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--fonts",    action="store_true", help="Dump all fonts found in PDF")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"{RED}Not found: {args.pdf}{RESET}")
        sys.exit(1)

    if args.fonts:
        print_fonts(args.pdf)
        return

    show_all = not (args.elements or args.chunks or args.compare)

    print(f"\n{BOLD}Parsing {args.pdf.name} with pdfplumber (generic)...{RESET}")
    print(f"  Heading detection: bold/medium weight font OR size > median*{RELATIVE_SIZE_MULTIPLIER}")
    print(f"  Space insertion:   gap > font_size * {SPACE_GAP_RATIO}")

    elements = parse_pdf_pdfplumber(args.pdf)
    chunks   = chunk_elements(elements)

    if args.elements or show_all:
        print_elements(elements)
    if args.chunks or show_all:
        print_chunks(chunks)
    if args.compare or show_all:
        print_comparison(args.pdf, chunks)


if __name__ == "__main__":
    main()
