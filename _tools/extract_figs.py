#!/usr/bin/env python3
"""从 arXiv 官方 HTML 提取论文图片，生成 figures.json 映射。

用法: python3 -I extract_figs.py <arxiv_id> <outdir>
  例: python3 -I extract_figs.py 2502.18965 /path/to/01_OneRec
产物:
  <outdir>/source.html        原始 HTML（供核对）
  <outdir>/fig/figN_i.<ext>   每张图片
  <outdir>/figures.json       [{"n": 图号, "caption": 英文原题, "files": [...]}, ...]
"""
import json
import os
import re
import sys
import urllib.request

from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    arxiv_id, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.join(outdir, "fig"), exist_ok=True)

    base = f"https://arxiv.org/html/{arxiv_id}"
    req = urllib.request.Request(base, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
        final_url = r.geturl()  # 例如 https://arxiv.org/html/2502.18965v1
    with open(os.path.join(outdir, "source.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML: {final_url} ({len(html)} chars)")

    soup = BeautifulSoup(html, "lxml")
    mapping = []
    # LaTeXML 的图环境：<figure class="ltx_figure">，含 <img class="ltx_graphics">
    # 与 <figcaption>（"Figure N: ..."）；兜底：任何含 img 且 caption 以 Figure 开头的 <figure>
    figures = soup.find_all("figure", class_="ltx_figure")
    if not figures:
        figures = [f for f in soup.find_all("figure")
                   if f.find("img") and (f.find("figcaption") is not None)]
    seen = set()
    for fig in figures:
        cap_el = fig.find("figcaption")
        if cap_el is None:
            continue
        cap_text = cap_el.get_text(" ", strip=True)
        m = re.match(r"^(?:Figure|Fig\.)\s*(\d+)", cap_text)
        if not m:
            continue
        n = int(m.group(1))
        # 图形可能以 <img src> / <object data=svg> / <embed src> 三种形式出现
        graphics = fig.find_all("img", src=True) + fig.find_all("object", data=True) \
            + fig.find_all("embed", src=True)
        files = []
        for i, g in enumerate(graphics):
            src = g.get("src") or g.get("data") or ""
            if src.startswith("data:") or src.startswith("/static/"):
                continue
            from urllib.parse import urljoin
            # arXiv HTML 的相对路径基准不定，逐个候选尝试
            candidates = [
                urljoin(final_url + "/", src),
                urljoin("https://arxiv.org/html/", src.lstrip("./")),
                urljoin(final_url, src),
            ]
            ext = os.path.splitext(src)[1].lstrip(".") or "png"
            name = f"fig{n}_{i}.{ext}" if len(graphics) > 1 else f"fig{n}.{ext}"
            path = os.path.join(outdir, "fig", name)
            for full in dict.fromkeys(candidates):
                try:
                    data = fetch(full)
                    with open(path, "wb") as f:
                        f.write(data)
                    files.append({"path": path,
                                  "w": g.get("width", ""), "h": g.get("height", "")})
                    print(f"  图{n}[{i}] <- {full} ({len(data)} bytes)")
                    break
                except Exception:  # noqa: BLE001
                    continue
            else:
                print(f"  下载失败 图{n}[{i}] 候选: {candidates}")
        if files and n not in seen:
            seen.add(n)
            mapping.append({"n": n, "caption": cap_text, "files": files})

    with open(os.path.join(outdir, "figures.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)
    print(f"共 {len(mapping)} 个图: {[m['n'] for m in mapping]}")


if __name__ == "__main__":
    main()
