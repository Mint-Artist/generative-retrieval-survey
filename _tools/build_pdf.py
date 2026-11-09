#!/usr/bin/env python3
"""把翻译 markdown + 提取的原文图片 排版为自包含 HTML，并用无头 Chrome 打印 PDF。

用法: python3 -I build_pdf.py --md <翻译.md> --figmap <figures.json> --out <输出.pdf> [--html <中间.html>]
依赖: pip install markdown pymupdf；本目录 katex/；/Applications/Google Chrome.app
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

MATH_TOKEN = "\u2b24MATH{}\u2b24"  # ●MATHi●


def to_html(text: str) -> str:
    """把 $$..$$ 与 $..$ 换成占位符（防 markdown 吞掉 _ ^ *），转 HTML 后还原为 KaTeX span。"""
    spans = []

    def stash(m):
        spans.append(m.group(0))  # 保留定界符，KaTeX 渲染时需要
        return MATH_TOKEN.format(len(spans) - 1)

    parts = re.split(r"(```.*?```)", text, flags=re.S)
    out = []
    for part in parts:
        if part.startswith("```"):
            out.append(part)
            continue
        part = re.sub(r"\$\$.+?\$\$", stash, part, flags=re.S)
        part = re.sub(r"(?<![\\$])\$[^\$\n]+?\$", stash, part)
        out.append(part)
    html = markdown.markdown("".join(out), extensions=["tables"])
    import html as _html
    return re.sub(
        MATH_TOKEN.format(r"(\d+)"),
        lambda m: '<span class="mtex">' + _html.escape(spans[int(m.group(1))], quote=False) + "</span>",
        html,
    )


def data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    if path.endswith(".svg"):
        mime = "image/svg+xml"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def inject_figures(html: str, figmap: list) -> tuple[str, list]:
    """把 <p>图N：…</p> 段落替换为 <figure>（原文图片 + 中英双语图题）。"""
    by_n = {int(m["n"]): m for m in figmap}
    used, missing = set(), []

    def repl(m):
        n = int(m.group(1))
        para = m.group(0)
        if n not in by_n:
            missing.append(n)
            return para
        used.add(n)
        info = by_n[n]
        imgs = ""
        for fp in info["files"]:
            if isinstance(fp, dict):
                style = ""
                if fp.get("w") and str(fp["w"]).isdigit():
                    style = f' style="width:{fp["w"]}px;height:auto"'
                imgs += f'<img src="{data_uri(fp["path"])}" alt="图{n}"{style}>'
            else:
                imgs += f'<img src="{data_uri(fp)}" alt="图{n}">'
        cn_cap = re.sub(r"^<p>\s*(?:<strong>)?\s*图\s*\d+\s*[：:]\s*(?:</strong>)?", "", para)
        cn_cap = re.sub(r"</p>$", "", cn_cap).strip()
        en_cap = info["caption"].strip()
        return (
            f"<figure>{imgs}"
            f'<figcaption><b>图{n}：</b>{cn_cap}'
            f'<span class="cap-en">{en_cap}</span></figcaption></figure>'
        )

    html = re.sub(r"<p>(?:<strong>)?\s*图\s*(\d+)\s*[：:].*?</p>", repl, html, flags=re.S)
    unused = sorted(set(by_n) - used)
    return html, missing + [f"(未使用){u}" for u in unused]


def katex_inline() -> str:
    css = open(os.path.join(HERE, "katex", "katex.min.css"), encoding="utf-8").read()
    css = re.sub(
        r"url\(fonts/([A-Za-z0-9_.-]+\.woff2)\)",
        lambda m: "url(" + data_uri(os.path.join(HERE, "katex", "fonts", m.group(1))) + ")",
        css,
    )
    js = open(os.path.join(HERE, "katex", "katex.min.js"), encoding="utf-8").read()
    ar = open(os.path.join(HERE, "katex", "auto-render.min.js"), encoding="utf-8").read()
    init = (
        "document.addEventListener('DOMContentLoaded',function(){"
        "renderMathInElement(document.body,{delimiters:["
        "{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],"
        "ignoredTags:['script','noscript','style','textarea'],"
        "throwOnError:false});});"
    )
    return f"<style>{css}</style><script>{js}</script><script>{ar}</script><script>{init}</script>"


CSS = """
@page { size: A4; margin: 2.2cm 2cm; }
* { box-sizing: border-box; }
body { font-family: "Songti SC","STSong","Noto Serif CJK SC","Source Han Serif SC",serif;
       font-size: 10.5pt; line-height: 1.78; color: #1a1a1a; margin: 0; }
h1,h2,h3,h4 { font-family: "PingFang SC","Heiti SC","Noto Sans CJK SC",sans-serif;
       line-height: 1.4; page-break-after: avoid; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 1em; }
h2 { font-size: 13.5pt; border-bottom: 1px solid #ccc; padding-bottom: .2em; margin-top: 1.6em; }
h3 { font-size: 12pt; margin-top: 1.3em; }
h4 { font-size: 11pt; }
p { margin: .5em 0; text-align: justify; }
strong { font-family: "PingFang SC","Heiti SC",sans-serif; }
hr { border: none; border-top: 1px solid #bbb; margin: 1.5em 0; }
table { border-collapse: collapse; margin: .8em auto; font-size: 9pt;
        page-break-inside: avoid; }
th,td { border: .5pt solid #888; padding: 3pt 6pt; text-align: center; }
th { background: #f0f0f0; font-family: "PingFang SC",sans-serif; }
figure { text-align: center; margin: 1.2em auto; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 22cm; margin: .3em auto; }
figcaption { font-size: 9pt; color: #333; margin-top: .3em; text-align: justify; }
.cap-en { display: block; color: #777; font-style: italic; font-size: 8.5pt; margin-top: .25em; }
code,pre { font-family: "SF Mono",Menlo,monospace; font-size: 9pt; }
pre { background: #f6f6f6; padding: .5em; border-radius: 4px; overflow-x: hidden;
      white-space: pre-wrap; page-break-inside: avoid; }
.katex { font-size: 1.02em; }
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--figmap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--html", default=None)
    args = ap.parse_args()
    for k in ("md", "figmap", "out"):
        setattr(args, k, os.path.abspath(getattr(args, k)))
    if args.html:
        args.html = os.path.abspath(args.html)

    md_text = open(args.md, encoding="utf-8").read()
    figmap = json.load(open(args.figmap, encoding="utf-8"))

    body = to_html(md_text)
    body, notes = inject_figures(body, figmap)
    html_doc = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<style>{CSS}</style>{katex_inline()}</head><body>{body}</body></html>"
    )
    html_path = args.html or (os.path.splitext(args.out)[0] + "_build.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    cmd = [
        CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--no-pdf-header-footer", "--virtual-time-budget=30000",
        f"--print-to-pdf={args.out}", f"file://{html_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not os.path.exists(args.out):
        sys.stderr.write((proc.stderr or "")[-2000:])
        raise SystemExit(f"Chrome 打印失败 returncode={proc.returncode}")

    import pymupdf
    doc = pymupdf.open(args.out)
    size = os.path.getsize(args.out)
    print(f"OK {args.out}: {doc.page_count} 页, {size/1e6:.1f} MB; html {len(html_doc)/1e6:.1f} MB")
    if notes:
        print(f"注意(图号缺失/未使用): {notes}")
        sys.exit(2 if any(isinstance(x, int) for x in notes) else 0)


if __name__ == "__main__":
    main()
