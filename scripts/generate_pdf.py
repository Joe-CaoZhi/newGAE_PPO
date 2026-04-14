#!/usr/bin/env python3
"""
Generate high-quality PDFs from Markdown paper drafts.
Handles: math equations (KaTeX), embedded images (base64), academic layout.
Usage:
    python3 scripts/generate_pdf.py
"""

import subprocess
import sys
import os
import re
import base64
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
FIGURES = ROOT / "results" / "paper_figures"

# ── CSS: A4 academic layout ──────────────────────────────────────────────────
ACADEMIC_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');

* { box-sizing: border-box; }

@page {
  size: A4;
  margin: 2.2cm 2cm 2.2cm 2cm;
}

html {
  font-size: 11pt;
  color: #111;
  background: white;
}

body {
  font-family: "Times New Roman", "Noto Serif SC", serif;
  line-height: 1.6;
  max-width: 100%;
  margin: 0;
  padding: 0;
}

/* Title */
h1.title {
  font-size: 16pt;
  font-weight: bold;
  text-align: center;
  margin-bottom: 0.3em;
  line-height: 1.3;
}

/* Section headings */
h2 {
  font-size: 13pt;
  font-weight: bold;
  margin-top: 1.4em;
  margin-bottom: 0.4em;
  border-bottom: 1px solid #ccc;
  padding-bottom: 0.1em;
}

h3 {
  font-size: 12pt;
  font-weight: bold;
  margin-top: 1.1em;
  margin-bottom: 0.3em;
}

h4 {
  font-size: 11pt;
  font-weight: bold;
  font-style: italic;
  margin-top: 0.9em;
  margin-bottom: 0.2em;
}

/* Paragraphs */
p {
  margin: 0.5em 0;
  text-align: justify;
}

/* Math blocks */
.math.display {
  overflow-x: auto;
  margin: 0.8em 2em;
  text-align: center;
}

/* Tables */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
  font-size: 9.5pt;
}

th {
  background: #f0f0f0;
  font-weight: bold;
  border: 1px solid #bbb;
  padding: 5px 8px;
  text-align: left;
}

td {
  border: 1px solid #bbb;
  padding: 4px 8px;
}

tr:nth-child(even) td {
  background: #fafafa;
}

/* Figures */
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1em auto;
}

/* Code blocks */
pre {
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 3px;
  padding: 0.7em 1em;
  font-size: 8.5pt;
  line-height: 1.4;
  overflow-x: auto;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: "Courier New", "Source Code Pro", monospace;
}

code {
  font-family: "Courier New", monospace;
  font-size: 90%;
  background: #f5f5f5;
  padding: 1px 3px;
  border-radius: 2px;
}

/* Blockquotes (notes/captions) */
blockquote {
  border-left: 3px solid #aaa;
  margin: 0.8em 0 0.8em 1em;
  padding: 0.3em 0.8em;
  color: #444;
  font-size: 9.5pt;
}

/* Horizontal rule */
hr {
  border: none;
  border-top: 1px solid #ccc;
  margin: 1.2em 0;
}

/* Lists */
ul, ol {
  margin: 0.4em 0 0.4em 1.5em;
  padding: 0;
}

li {
  margin: 0.2em 0;
}

/* Abstract box */
#abstract, section.abstract {
  border: 1px solid #ccc;
  padding: 0.6em 1em;
  margin: 1em 0;
  background: #fafafa;
  font-size: 10pt;
}

/* Figure captions */
em {
  font-style: italic;
}

/* Strong */
strong {
  font-weight: bold;
}

/* Page break hints */
h2, h3 {
  page-break-after: avoid;
}

pre, table {
  page-break-inside: avoid;
}

img {
  page-break-inside: avoid;
}
"""

KATEX_CACHE_DIR = Path("/tmp/katex_local")
KATEX_CDN_BASE  = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/"
KATEX_FONTS = [
    "KaTeX_AMS-Regular.woff2", "KaTeX_Caligraphic-Bold.woff2",
    "KaTeX_Caligraphic-Regular.woff2", "KaTeX_Fraktur-Bold.woff2",
    "KaTeX_Fraktur-Regular.woff2", "KaTeX_Main-Bold.woff2",
    "KaTeX_Main-BoldItalic.woff2", "KaTeX_Main-Italic.woff2",
    "KaTeX_Main-Regular.woff2", "KaTeX_Math-BoldItalic.woff2",
    "KaTeX_Math-Italic.woff2", "KaTeX_SansSerif-Bold.woff2",
    "KaTeX_SansSerif-Italic.woff2", "KaTeX_SansSerif-Regular.woff2",
    "KaTeX_Script-Regular.woff2", "KaTeX_Size1-Regular.woff2",
    "KaTeX_Size2-Regular.woff2", "KaTeX_Size3-Regular.woff2",
    "KaTeX_Size4-Regular.woff2", "KaTeX_Typewriter-Regular.woff2",
]


def _fetch(url: str, dest: Path):
    """Download url → dest if not already cached."""
    if dest.exists():
        return
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _build_katex_header() -> str:
    """
    Build a fully self-contained <style>+<script> block with KaTeX inlined.
    Downloads KaTeX assets to /tmp/katex_local/ on first run (cached thereafter).
    """
    print("  Loading KaTeX assets (offline) ...")
    cache = KATEX_CACHE_DIR

    # ── 1. KaTeX JS ────────────────────────────────────────────────────────
    katex_js_path      = cache / "katex.min.js"
    autorender_js_path = cache / "auto-render.min.js"
    _fetch(KATEX_CDN_BASE + "katex.min.js",                       katex_js_path)
    _fetch(KATEX_CDN_BASE + "contrib/auto-render.min.js",         autorender_js_path)

    # ── 2. KaTeX fonts ─────────────────────────────────────────────────────
    fonts_dir = cache / "fonts"
    for fname in KATEX_FONTS:
        _fetch(KATEX_CDN_BASE + "fonts/" + fname, fonts_dir / fname)

    # ── 3. KaTeX CSS – patch font URLs → base64 ────────────────────────────
    katex_css_path = cache / "katex.min.css"
    _fetch(KATEX_CDN_BASE + "katex.min.css", katex_css_path)

    css_text = katex_css_path.read_text(encoding="utf-8")

    def embed_font(m):
        fname = m.group(1)
        fpath = fonts_dir / fname
        if fpath.exists():
            return f"url(data:font/woff2;base64,{_b64(fpath)})"
        return m.group(0)

    css_inline = re.sub(r"url\(fonts/([^)]+\.woff2)\)", embed_font, css_text)
    # Remove remaining non-woff2 @font-face src entries so browser picks woff2
    css_inline = re.sub(r"url\(fonts/[^)]+\.(?:woff|ttf)\)[^,;]*[,]?", "", css_inline)

    katex_js   = katex_js_path.read_text(encoding="utf-8")
    autorender = autorender_js_path.read_text(encoding="utf-8")

    # The auto-render script is appended to <body> via a defer inline block
    # so it runs after the DOM is fully parsed.
    header = f"""<style>
{css_inline}
</style>
<script>
{katex_js}
</script>
<script>
{autorender}
</script>
<script>
// Run immediately – KaTeX and auto-render are already loaded synchronously above
(function() {{
  function doRender() {{
    renderMathInElement(document.body, {{
      delimiters: [
        {{left: "$$",  right: "$$",  display: true}},
        {{left: "$",   right: "$",   display: false}},
        {{left: "\\\\(", right: "\\\\)", display: false}},
        {{left: "\\\\[", right: "\\\\]", display: true}}
      ],
      throwOnError: false,
      strict: false
    }});
    document.querySelectorAll(".math.display").forEach(function(el) {{
      el.style.textAlign = "center";
      el.style.margin = "0.8em 0";
    }});
  }}
  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", doRender);
  }} else {{
    doRender();
  }}
}})();
</script>
"""
    return header


def img_to_base64(img_path: Path) -> str:
    """Convert image file to base64 data URI."""
    ext = img_path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(ext, "png")
    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def embed_images(html: str, md_dir: Path) -> str:
    """Replace all img src with base64-embedded versions."""
    def replace_src(m):
        src = m.group(1)
        # Skip already-embedded data URIs
        if src.startswith("data:"):
            return m.group(0)
        # Resolve relative path from the markdown file's directory
        candidates = [
            md_dir / src,
            md_dir / Path(src).name,
            FIGURES / Path(src).name,
            Path(src),
        ]
        for p in candidates:
            if p.exists():
                try:
                    return f'src="{img_to_base64(p)}"'
                except Exception as e:
                    print(f"  [warn] Could not embed {p}: {e}")
                    break
        print(f"  [warn] Image not found: {src}")
        return m.group(0)  # keep original if not found

    # Only replace src inside <img ...> tags, not script/link tags
    def replace_img_tag(m):
        return re.sub(r'src="([^"]+)"', replace_src, m.group(0))

    return re.sub(r'<img\b[^>]*>', replace_img_tag, html)


def build_html(md_path: Path, title: str) -> str:
    """Convert Markdown to styled HTML with embedded images and KaTeX math."""
    print(f"  Converting {md_path.name} → HTML ...")

    # pandoc: output HTML with MathJax markers (we'll swap for KaTeX)
    result = subprocess.run(
        [
            "pandoc",
            str(md_path),
            "--standalone",
            "--mathjax",          # marks math with class="math display/inline"
            f"--metadata=title:{title}",
            "--to=html5",
            "-",
        ],
        capture_output=True, text=True, check=True
    )
    html = result.stdout

    # Inject KaTeX (replaces MathJax script tags)
    html = re.sub(
        r'<script[^>]+mathjax[^>]*>.*?</script>',
        "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(
        r'https://cdn\.jsdelivr\.net/npm/mathjax[^"]*"[^>]*>',
        "", html
    )
    # Remove all MathJax script tags
    html = re.sub(
        r'<script[^>]*MathJax[^>]*>.*?</script>',
        "", html, flags=re.DOTALL | re.IGNORECASE
    )

    # Inject our CSS and KaTeX into <head>
    # Use a lambda to avoid re.sub interpreting backslashes in KaTeX JS
    katex_header = _build_katex_header()
    injection = f"<style>\n{ACADEMIC_CSS}\n</style>\n{katex_header}"
    html = re.sub(r"</head>", lambda m: injection + "</head>", html)

    # Embed images as base64
    html = embed_images(html, md_path.parent)

    return html


def html_to_pdf(html: str, out_pdf: Path):
    """Use Chrome headless to convert HTML → PDF, waiting for KaTeX to render."""
    # Write HTML to a temp file (Chrome needs a real file path for local resources)
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w",
                                     encoding="utf-8", delete=False) as f:
        f.write(html)
        tmp_html = f.name

    print(f"  Chrome headless → {out_pdf.name} ...")
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={out_pdf}",
        f"file://{tmp_html}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(tmp_html)

    if out_pdf.exists():
        size_kb = out_pdf.stat().st_size // 1024
        print(f"  ✓ {out_pdf.name}  ({size_kb} KB)")
    else:
        print(f"  ✗ Failed to generate {out_pdf.name}")
        print(result.stderr[-500:] if result.stderr else "(no stderr)")
        sys.exit(1)


def main():
    print("=== HCGAE Paper PDF Generator ===\n")

    tasks = [
        {
            "md": DOCS / "paper_draft.md",
            "title": "HCGAE: Hindsight-Corrected Generalized Advantage Estimation",
            "out": DOCS / "paper_draft.pdf",
        },
        {
            "md": DOCS / "paper_draft_zh.md",
            "title": "HCGAE：事后修正广义优势估计",
            "out": DOCS / "paper_draft_zh.pdf",
        },
    ]

    for t in tasks:
        print(f"[{t['md'].name}]")
        html = build_html(t["md"], t["title"])
        html_to_pdf(html, t["out"])
        print()

    print("Done! PDFs saved to docs/")


if __name__ == "__main__":
    main()

