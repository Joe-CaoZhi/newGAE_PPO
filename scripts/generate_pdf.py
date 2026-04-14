#!/usr/bin/env python3
"""
Generate high-quality PDFs from Markdown paper drafts.
Handles: math equations (KaTeX via CDP), embedded images (base64), academic layout.

Strategy:
  1. pandoc  Markdown → HTML (--mathjax marks math with class="math display/inline")
  2. Inject KaTeX (offline, base64-embedded fonts) + academic CSS
  3. Launch Chrome with --remote-debugging-port (CDP mode)
  4. Open the HTML page via CDP, wait for KaTeX to finish rendering
  5. Call Page.printToPDF via CDP → base64 PDF → write to file

This avoids the problem of Chrome headless --print-to-pdf not waiting for JS.
"""

import subprocess
import sys
import os
import re
import base64
import tempfile
import time
import json
import socket
import threading
from pathlib import Path

ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"
FIGURES = ROOT / "results" / "paper_figures"

# ── CSS: A4 academic layout ──────────────────────────────────────────────────
ACADEMIC_CSS = """
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
  font-family: "Times New Roman", "Georgia", serif;
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
.math.display, .katex-display {
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
  font-family: "Courier New", monospace;
}

code {
  font-family: "Courier New", monospace;
  font-size: 90%;
  background: #f5f5f5;
  padding: 1px 3px;
  border-radius: 2px;
}

/* Blockquotes */
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

/* KaTeX display alignment */
.katex-display > .katex {
  text-align: center;
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
    print(f"    Downloading {dest.name} ...")
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
    _fetch(KATEX_CDN_BASE + "katex.min.js",                   katex_js_path)
    _fetch(KATEX_CDN_BASE + "contrib/auto-render.min.js",     autorender_js_path)

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
    css_inline = re.sub(r"url\(fonts/[^)]+\.(?:woff|ttf)\)[^,;]*[,]?", "", css_inline)

    katex_js   = katex_js_path.read_text(encoding="utf-8")
    autorender = autorender_js_path.read_text(encoding="utf-8")

    # Inject KaTeX + auto-render + trigger render immediately (synchronous)
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
// Signal flag: set to true once KaTeX rendering is complete
window.__katex_done = false;

function _doKatexRender() {{
  try {{
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
    document.querySelectorAll(".math.display, .katex-display").forEach(function(el) {{
      el.style.textAlign = "center";
      el.style.margin = "0.8em 0";
    }});
  }} catch(e) {{
    console.error("KaTeX render error:", e);
  }}
  window.__katex_done = true;
}}

if (document.readyState === "loading") {{
  document.addEventListener("DOMContentLoaded", _doKatexRender);
}} else {{
  _doKatexRender();
}}
</script>
"""
    return header


def img_to_base64(img_path: Path) -> str:
    """Convert image file to base64 data URI."""
    ext = img_path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
            "gif": "gif", "webp": "webp"}.get(ext, "png")
    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def embed_images(html: str, md_dir: Path) -> str:
    """Replace all img src with base64-embedded versions."""
    def replace_src(m):
        src = m.group(1)
        if src.startswith("data:"):
            return m.group(0)
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
        return m.group(0)

    def replace_img_tag(m):
        return re.sub(r'src="([^"]+)"', replace_src, m.group(0))

    return re.sub(r'<img\b[^>]*>', replace_img_tag, html)


def build_html(md_path: Path, title: str) -> str:
    """Convert Markdown to styled HTML with embedded images and KaTeX math."""
    print(f"  Converting {md_path.name} → HTML ...")

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

    # Remove MathJax script tags (we replace with KaTeX)
    html = re.sub(
        r'<script[^>]+mathjax[^>]*>.*?</script>',
        "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(
        r'https://cdn\.jsdelivr\.net/npm/mathjax[^"]*"[^>]*>',
        "", html
    )
    html = re.sub(
        r'<script[^>]*MathJax[^>]*>.*?</script>',
        "", html, flags=re.DOTALL | re.IGNORECASE
    )

    # Inject academic CSS and KaTeX into <head>
    katex_header = _build_katex_header()
    injection = f"<style>\n{ACADEMIC_CSS}\n</style>\n{katex_header}"
    html = re.sub(r"</head>", lambda m: injection + "</head>", html)

    # Embed images as base64
    html = embed_images(html, md_path.parent)

    return html


# ── CDP-based PDF generation ──────────────────────────────────────────────────

def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _cdp_request(port: int, method: str, params: dict = None, timeout: float = 60) -> dict:
    """Make a single CDP JSON RPC call via HTTP."""
    import urllib.request as ur
    import urllib.error

    payload = json.dumps({"id": 1, "method": method,
                          "params": params or {}}).encode()
    url = f"http://127.0.0.1:{port}/json/new"

    # First get the WebSocket debugger URL for the page
    deadline = time.time() + timeout
    ws_url = None
    while time.time() < deadline:
        try:
            with ur.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as r:
                pages = json.loads(r.read())
                for p in pages:
                    if p.get("type") == "page":
                        ws_url = p.get("webSocketDebuggerUrl")
                        break
            if ws_url:
                break
        except Exception:
            time.sleep(0.3)

    return ws_url


def html_to_pdf_cdp(html: str, out_pdf: Path):
    """
    Use Chrome DevTools Protocol (CDP) to:
      1. Start Chrome with --remote-debugging-port
      2. Navigate to the HTML page
      3. Poll until window.__katex_done == true
      4. Call Page.printToPDF
    """
    import urllib.request as ur
    import urllib.error

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    port   = _find_free_port()

    # Write HTML to temp file
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w",
                                     encoding="utf-8", delete=False) as f:
        f.write(html)
        tmp_html = f.name
    file_url = f"file://{tmp_html}"

    print(f"  Starting Chrome CDP on port {port} ...")
    proc = subprocess.Popen(
        [
            chrome,
            f"--remote-debugging-port={port}",
            "--headless=new",      # new headless supports Page.printToPDF on page targets
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--disable-default-apps",
            "--disable-extensions",
            "--disable-background-networking",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        # Wait for Chrome to start
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                with ur.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1):
                    break
            except Exception:
                time.sleep(0.3)
        else:
            raise RuntimeError("Chrome did not start in time")

        # Get the first available "page" type tab (not background_page)
        with ur.urlopen(f"http://127.0.0.1:{port}/json", timeout=5) as r:
            pages = json.loads(r.read())
        page = None
        for p in pages:
            if p.get("type") == "page":
                page = p
                break
        if page is None:
            raise RuntimeError(f"No page target found. Available: {[p.get('type') for p in pages]}")
        ws_url = page["webSocketDebuggerUrl"]
        print(f"  CDP connected: {ws_url[:60]}...")

        # Use websocket to talk CDP
        pdf_b64 = _cdp_websocket_pdf(ws_url, file_url, port)

        pdf_bytes = base64.b64decode(pdf_b64)
        out_pdf.write_bytes(pdf_bytes)
        size_kb = len(pdf_bytes) // 1024
        print(f"  ✓ {out_pdf.name}  ({size_kb} KB)")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            os.unlink(tmp_html)
        except Exception:
            pass


def _cdp_websocket_pdf(ws_url: str, file_url: str, port: int) -> str:
    """
    Use Python's built-in http + manual WebSocket to drive CDP.
    Returns the base64-encoded PDF string.
    """
    import urllib.parse
    import struct
    import hashlib
    import ssl

    # Parse ws URL
    parsed = urllib.parse.urlparse(ws_url)
    host = parsed.hostname
    ws_port = parsed.port or 80
    path = parsed.path

    # WebSocket handshake
    key_bytes = base64.b64encode(os.urandom(16))
    key = key_bytes.decode("ascii")

    sock = socket.create_connection((host, ws_port), timeout=30)

    handshake = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{ws_port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(handshake.encode())

    # Read HTTP response headers
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += sock.recv(4096)
    # Should be 101 Switching Protocols

    def ws_send(sock, data: dict):
        msg = json.dumps(data).encode("utf-8")
        length = len(msg)
        mask_key = os.urandom(4)
        masked = bytes(b ^ mask_key[i % 4] for i, b in enumerate(msg))
        if length < 126:
            header = bytes([0x81, 0x80 | length]) + mask_key
        elif length < 65536:
            header = bytes([0x81, 0xFE]) + struct.pack(">H", length) + mask_key
        else:
            header = bytes([0x81, 0xFF]) + struct.pack(">Q", length) + mask_key
        sock.sendall(header + masked)

    def ws_recv(sock) -> dict:
        """Receive one complete WebSocket text frame."""
        # Read 2-byte header
        header = b""
        while len(header) < 2:
            header += sock.recv(2 - len(header))
        b0, b1 = header[0], header[1]
        masked = (b1 & 0x80) != 0
        length = b1 & 0x7F
        if length == 126:
            ext = b""
            while len(ext) < 2:
                ext += sock.recv(2 - len(ext))
            length = struct.unpack(">H", ext)[0]
        elif length == 127:
            ext = b""
            while len(ext) < 8:
                ext += sock.recv(8 - len(ext))
            length = struct.unpack(">Q", ext)[0]
        if masked:
            mask = b""
            while len(mask) < 4:
                mask += sock.recv(4 - len(mask))
        payload = b""
        while len(payload) < length:
            chunk = sock.recv(min(65536, length - len(payload)))
            if not chunk:
                break
            payload += chunk
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return json.loads(payload.decode("utf-8"))

    msg_id = [0]
    def send(method, params=None):
        msg_id[0] += 1
        ws_send(sock, {"id": msg_id[0], "method": method, "params": params or {}})
        return msg_id[0]

    sock.settimeout(120)

    # Enable Page domain
    send("Page.enable")
    _resp = ws_recv(sock)  # ack

    # Navigate to the file
    nav_id = send("Page.navigate", {"url": file_url})
    print(f"  Navigating to {file_url[:60]}...")

    # Wait for Page.loadEventFired
    deadline = time.time() + 60
    while time.time() < deadline:
        msg = ws_recv(sock)
        if msg.get("method") == "Page.loadEventFired":
            print("  Page load event fired.")
            break
        if msg.get("id") == nav_id and "error" in msg:
            raise RuntimeError(f"Navigation error: {msg['error']}")

    # Now poll until window.__katex_done == true
    print("  Waiting for KaTeX rendering ...")
    eval_deadline = time.time() + 30
    while time.time() < eval_deadline:
        eval_id = send("Runtime.evaluate", {
            "expression": "window.__katex_done === true",
            "returnByValue": True
        })
        # Drain messages until we get our response
        t_start = time.time()
        while time.time() - t_start < 5:
            msg = ws_recv(sock)
            if msg.get("id") == eval_id:
                result = msg.get("result", {}).get("result", {})
                if result.get("value") is True:
                    print("  KaTeX rendering complete.")
                    break
                break
        else:
            continue
        if result.get("value") is True:
            break
        time.sleep(0.3)

    # Extra safety wait to ensure any remaining layout/repaint is done
    time.sleep(1.0)

    # Print to PDF
    print("  Printing to PDF ...")
    pdf_id = send("Page.printToPDF", {
        "printBackground": True,
        "paperWidth":  8.27,    # A4 in inches
        "paperHeight": 11.69,
        "marginTop":    0.87,   # 2.2cm
        "marginBottom": 0.87,
        "marginLeft":   0.79,   # 2cm
        "marginRight":  0.79,
        "displayHeaderFooter": False,
        "preferCSSPageSize": True,
    })

    deadline = time.time() + 120
    while time.time() < deadline:
        msg = ws_recv(sock)
        if msg.get("id") == pdf_id:
            if "error" in msg:
                raise RuntimeError(f"PDF error: {msg['error']}")
            return msg["result"]["data"]

    raise RuntimeError("Timeout waiting for PDF")


def html_to_pdf(html: str, out_pdf: Path):
    """Entry point: use CDP if available, else fallback."""
    html_to_pdf_cdp(html, out_pdf)


def build_html(md_path: Path, title: str) -> str:
    """Convert Markdown to styled HTML with embedded images and KaTeX math."""
    print(f"  Converting {md_path.name} → HTML ...")

    result = subprocess.run(
        [
            "pandoc",
            str(md_path),
            "--standalone",
            "--mathjax",
            f"--metadata=title:{title}",
            "--to=html5",
            "-",
        ],
        capture_output=True, text=True, check=True
    )
    html = result.stdout

    # Remove MathJax script tags
    html = re.sub(
        r'<script[^>]+mathjax[^>]*>.*?</script>',
        "", html, flags=re.DOTALL | re.IGNORECASE
    )
    html = re.sub(
        r'https://cdn\.jsdelivr\.net/npm/mathjax[^"]*"[^>]*>',
        "", html
    )
    html = re.sub(
        r'<script[^>]*MathJax[^>]*>.*?</script>',
        "", html, flags=re.DOTALL | re.IGNORECASE
    )

    # Inject academic CSS and KaTeX
    katex_header = _build_katex_header()
    injection = f"<style>\n{ACADEMIC_CSS}\n</style>\n{katex_header}"
    html = re.sub(r"</head>", lambda m: injection + "</head>", html)

    # Embed images as base64
    html = embed_images(html, md_path.parent)

    return html


def main():
    print("=== HCGAE Paper PDF Generator (CDP mode) ===\n")

    tasks = [
        {
            "md":    DOCS / "paper_draft.md",
            "title": "HCGAE: Hindsight-Corrected Generalized Advantage Estimation",
            "out":   DOCS / "paper_draft.pdf",
        },
        {
            "md":    DOCS / "paper_draft_zh.md",
            "title": "HCGAE：事后修正广义优势估计",
            "out":   DOCS / "paper_draft_zh.pdf",
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

