#!/usr/bin/env python3
"""polish_cjk.py — 双语 EPUB 成品的中文排版后处理（通用，非某本书专用）。

本仓库产出的双语书里，译文块和原文块是同一个父元素下的兄弟节点，
译文块通常没有任何 class / lang 可以区分，于是：

  ① 译文继承了原书给拉丁文字设计的 font-family（Verdana / Arial 之类），
     这些字体没有中文字形，每个汉字都逐字符回退到阅读器随机挑的系统字体
     —— 字重对不上（偏淡）、汉字撑满 em 框（偏大）、中英基线不齐（排版乱）。
  ② 原书正文一般不设 line-height，中文按拉丁文的默认行距排会挤。
  ③ 翻译器的分段映射照搬了英文的空格位置，译文块首尾和行内标签内侧
     留下大量游离空格（`<em> 《书名》 </em>`）。

本模块把这三条一次修掉：

  1. 识别以中日韩文字为主的**叶子块级元素**，给它加 lang/xml:lang="zh-CN"
     和 class="zh-translation"；
  2. 往每个文档的 <head> 末尾注入一段带 id 的 <style>（幂等，重复跑只会
     整段替换，不会叠加）；
  3. 只在被判定为中文的块里清理游离空格；
  4. （默认开）把一份**只含本书用字的中文字体子集**打进 EPUB：注册到 OPF
     manifest，在注入的 CSS 里加 `@font-face`，并把这个字族排进字体栈的
     中文位置（拉丁字体仍在最前，译文里夹的英文词和数字继续走拉丁字形）。
     这样渲染不再取决于"阅读器碰巧装了什么中文字体"。

**安全性**：不做 DOM 往返序列化。解析只用来定位偏移量，改动以定点替换的
方式打在原始文本上，所以没被判定为中文的部分（英文原文、<pre>/<code> 里
被 mask_code 逐字节还原过的代码、图片、toc.ncx、OPF）保证逐字节不变。
重打包时按源 zip 的条目顺序与时间戳逐条重写，mimetype 仍是第一个条目且
不压缩（与 pdf_to_epub._repackage_epub 的约束一致），因此输出也是确定性的。

字体子集用 fontTools 现做，源字体默认是系统里的 Noto Sans CJK SC
（OFL 许可，可自由内嵌分发）。两档覆盖范围：`book` 只装本书译文实际用到的
字，`gb2312` 再并上 GB2312 全量（读者在阅读器里加批注也不掉字体，代价是
约 1.2 MB）。字体源不可用时**明确报错**，不会静默退化成"没嵌字体"。

用法：
    uv run python polish_cjk.py IN.epub OUT.epub          # 处理（含内嵌字体）
    uv run python polish_cjk.py --stats IN.epub           # 只统计，不写文件
    uv run python polish_cjk.py IN.epub OUT.epub --no-embed-font
    uv run python polish_cjk.py IN.epub OUT.epub --font-coverage gb2312
    uv run python polish_cjk.py IN.epub OUT.epub --line-height 1.8 --threshold 0.4
"""

from __future__ import annotations

import argparse
import collections
import html
import io
import posixpath
import re
import sys
import zipfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

# ── 常量 ────────────────────────────────────────────────────────────────

CLASS_NAME = "zh-translation"
STYLE_ID = "zh-polish"

# 块级元素：判定与打标的候选。<pre> 在列表里，是为了让任何包含代码清单的
# 容器都不会被当成"叶子块"，从而永远不会被打标或改动。
BLOCK_TAGS = frozenset("""
    p div li h1 h2 h3 h4 h5 h6 blockquote td th dd dt caption figcaption
    pre section article aside header footer nav main ul ol dl table tr
    thead tbody tfoot figure address hgroup details summary body html
""".split())

# 这些元素内部的文本一个字符都不碰，判定中文占比时也不计入。
OPAQUE_TAGS = frozenset("pre code kbd samp script style var tt math svg".split())

VOID_TAGS = frozenset("""
    area base br col embed hr img input link meta param source track wbr
""".split())

SENTINEL = "￼"  # 代表"不可见/不可改的一段内容"（实体、图片、代码）

_WS = " \t\r\n\f\v"

# 默认字体栈：拉丁字体在前（保证块内的英文词、数字仍用无衬线拉丁字形），
# 中文字体在后逐级回退，覆盖 macOS / Windows / Linux / 各家阅读器。
LATIN_FONTS = '-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial'

CJK_FONTS = (
    '"PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", '
    '"Source Han Sans SC", "Source Han Sans CN", "Noto Sans CJK SC", '
    '"Noto Sans SC", "WenQuanYi Micro Hei", "Heiti SC", "STHeiti", '
    '"SimHei"'
)

DEFAULT_FONT_STACK = f"{LATIN_FONTS}, {CJK_FONTS}, sans-serif"

DEFAULT_MONO_STACK = (
    '"JetBrains Mono", Consolas, "Liberation Mono", Menlo, "Courier New", monospace'
)

# ── 内嵌字体 ────────────────────────────────────────────────────────────

# 字族名带上仓库前缀，避免和阅读器/系统里任何真实字体撞名（撞名会让阅读器
# 用系统那份，内嵌子集就白搭了）。
EMBED_FAMILY = "EpubTranslatorZhSans"
EMBED_FILE_STEM = "EpubTranslatorZhSans"
FONT_ITEM_ID = "zh-polish-font"
FONT_DIR_FALLBACK = "Misc"

# ttc 里 0=JP 1=KR 2=SC 3=TC 4=HK；中文正文要 SC 的字形（"直"、"骨"等
# 汉字的 JP/SC 写法不同），所以默认取 2。
DEFAULT_FONT_FILE = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
DEFAULT_FONT_NUMBER = 2

FONT_MEDIA_TYPES = {
    ".woff": ("font/woff", "woff"),
    ".woff2": ("font/woff2", "woff2"),
    ".otf": ("font/otf", "opentype"),
    ".ttf": ("font/ttf", "truetype"),
}

# 新增 zip 条目用固定时间戳（zip 的最早可表示时间），保证重复运行的输出
# 逐字节相同。
FONT_EPOCH = (1980, 1, 1, 0, 0, 0)

# 字体栈里"这一项是中文字体"的判据：内嵌字族要插在第一个中文字体之前。
_CJK_FAMILY_RE = re.compile(
    r"PingFang|Hiragino|YaHei|雅黑|Source Han|Noto Sans CJK|Noto Serif CJK|"
    r"Noto Sans SC|Noto Sans TC|WenQuanYi|Heiti|SimHei|SimSun|黑体|宋体|"
    r"FangSong|KaiTi|楷体|仿宋|Song|Ming|Kai|CJK",
    re.I,
)
_GENERIC_FAMILIES = frozenset(
    "serif sans-serif monospace cursive fantasy system-ui".split()
)


def is_cjk_script(ch: str) -> bool:
    """真正的中日韩文字（判定"这个块是不是译文"用，不含标点）。"""
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF      # 假名
        or 0x3400 <= o <= 0x4DBF   # 扩展 A
        or 0x4E00 <= o <= 0x9FFF   # 基本区
        or 0xAC00 <= o <= 0xD7AF   # 谚文
        or 0xF900 <= o <= 0xFAFF   # 兼容表意
        or 0x20000 <= o <= 0x3FFFF # 扩展 B+
    )


def is_cjk_punct(ch: str) -> bool:
    """中日韩标点与全角形式：它们自带左右留白，旁边的空格一律是多余的。"""
    return is_cjk_wide(ch) and not is_cjk_script(ch)


def is_cjk_wide(ch: str) -> bool:
    """文字 + 中日韩标点 + 全角形式（清理空格时用）。"""
    o = ord(ch)
    return (
        is_cjk_script(ch)
        or 0x3000 <= o <= 0x303F   # 、。《》「」…
        or 0xFE10 <= o <= 0xFE6F   # 竖排/兼容标点
        or 0xFF01 <= o <= 0xFF60   # 全角 ！（）：，等
        or 0xFFE0 <= o <= 0xFFE6
    )


# ── 解析：只为了拿偏移量 ─────────────────────────────────────────────────


class _Node:
    __slots__ = ("tag", "raw_start", "raw_end", "end_start", "end_end", "children")

    def __init__(self, tag: str, raw_start: int, raw_end: int) -> None:
        self.tag = tag
        self.raw_start = raw_start      # 开始标签 '<' 的偏移
        self.raw_end = raw_end          # 开始标签 '>' 之后的偏移
        self.end_start: int | None = None   # 结束标签 '<' 的偏移
        self.end_end: int | None = None
        self.children: list[tuple[str, object]] = []  # ("text",(s,e)) | ("opaque",(s,e)) | ("node",_Node)


class _Tree(HTMLParser):
    """把文档解析成带源码偏移的树。不重新序列化任何东西。"""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=False)
        self.text = text
        self.line_starts = [0]
        for m in re.finditer("\n", text):
            self.line_starts.append(m.end())
        self.root = _Node("#root", 0, 0)
        self.stack = [self.root]
        self.feed(text)
        self.close()

    def _off(self) -> int:
        line, col = self.getpos()
        return self.line_starts[line - 1] + col

    # --- 事件 ---

    def handle_starttag(self, tag, attrs):
        start = self._off()
        raw = self.get_starttag_text() or f"<{tag}>"
        node = _Node(tag, start, start + len(raw))
        self.stack[-1].children.append(("node", node))
        if tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        start = self._off()
        raw = self.get_starttag_text() or f"<{tag}/>"
        node = _Node(tag, start, start + len(raw))
        node.end_start = node.end_end = start + len(raw)
        self.stack[-1].children.append(("node", node))

    def handle_endtag(self, tag):
        start = self._off()
        gt = self.text.find(">", start)
        end = gt + 1 if gt != -1 else start
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                self.stack[i].end_start = start
                self.stack[i].end_end = end
                del self.stack[i:]
                return
        # 没有匹配的开始标签：忽略

    def handle_data(self, data):
        if not data:
            return
        start = self._off()
        self.stack[-1].children.append(("text", (start, start + len(data))))

    def _opaque_run(self, pattern: str) -> None:
        start = self._off()
        m = re.compile(pattern).match(self.text, start)
        end = m.end() if m else start + 1
        self.stack[-1].children.append(("opaque", (start, end)))

    def handle_entityref(self, name):
        self._opaque_run(r"&\w+;?")

    def handle_charref(self, name):
        self._opaque_run(r"&#[xX]?[0-9a-fA-F]+;?")

    def handle_comment(self, data):
        self._opaque_run(r"<!--.*?-->")


def _walk(node: _Node):
    yield node
    for kind, child in node.children:
        if kind == "node":
            yield from _walk(child)


def _stream(node: _Node, text: str) -> list[tuple[str, int | None]]:
    """块内的"可见字符流"：行内标签透明，代码/实体/图片折成一个哨兵字符。"""
    out: list[tuple[str, int | None]] = []

    def walk(n: _Node) -> None:
        for kind, child in n.children:
            if kind == "text":
                s, e = child  # type: ignore[misc]
                for i in range(s, e):
                    out.append((text[i], i))
            elif kind == "opaque":
                out.append((SENTINEL, None))
            else:
                c: _Node = child  # type: ignore[assignment]
                if c.tag in OPAQUE_TAGS or c.tag in VOID_TAGS or c.tag in BLOCK_TAGS:
                    out.append((SENTINEL, None))
                else:
                    walk(c)

    walk(node)
    return out


# ── 判定与改写 ──────────────────────────────────────────────────────────


# 一个汉字大致抵两个拉丁字母的信息量与视觉宽度，判定占比时给 CJK 加权，
# 否则"使用 PyTorch"这种短标题会被算成英文。
CJK_WEIGHT = 2.0


def _is_cjk_block(stream, threshold: float, min_cjk: int) -> bool:
    cjk = sum(1 for ch, _ in stream if is_cjk_script(ch))
    if cjk < min_cjk:
        return False
    latin = sum(1 for ch, _ in stream if ch.isascii() and ch.isalpha())
    w = cjk * CJK_WEIGHT
    return w / (w + latin) >= threshold


def _mark_starttag(raw: str) -> str | None:
    """给开始标签加 class / lang。已经标过则返回 None。"""
    m = re.match(r"<([A-Za-z][-\w:]*)((?:.|\n)*?)(/?)>\Z", raw)
    if not m:
        return None
    tag, attrs, slash = m.group(1), m.group(2), m.group(3)
    cm = re.search(r"""(\bclass\s*=\s*)(["'])(.*?)\2""", attrs, re.S)
    if cm:
        classes = cm.group(3).split()
        if CLASS_NAME in classes:
            return None  # 幂等
        new_val = (cm.group(3) + " " + CLASS_NAME).strip()
        attrs = attrs[: cm.start(3)] + new_val + attrs[cm.end(3) :]
    else:
        attrs = attrs.rstrip() + f' class="{CLASS_NAME}"'
    if not re.search(r"\blang\s*=", attrs):
        attrs += ' lang="zh-CN" xml:lang="zh-CN"'
    return f"<{tag}{attrs}{slash}>"


def _space_edits(stream) -> list[tuple[int, int, str]]:
    """块内游离空格的定点编辑。返回 (start, end, replacement)。"""
    edits: list[tuple[int, int, str]] = []
    n = len(stream)
    i = 0
    while i < n:
        if stream[i][0] not in _WS:
            i += 1
            continue
        j = i
        while j < n and stream[j][0] in _WS:
            j += 1
        prev_ch = stream[i - 1][0] if i > 0 else None
        next_ch = stream[j][0] if j < n else None
        run = stream[i:j]

        drop = (
            prev_ch is None                      # 块首
            or next_ch is None                   # 块尾
            or (is_cjk_wide(prev_ch) and is_cjk_wide(next_ch))  # 两个中文之间
            or is_cjk_punct(prev_ch) or is_cjk_punct(next_ch)   # 中文标点自带留白
        )
        if drop:
            keep = ""
        elif (is_cjk_wide(prev_ch) or is_cjk_wide(next_ch)) and len(run) > 1:
            keep = " "                           # 中文一侧的多余空白折成一个
        else:
            i = j
            continue

        offs = [o for _, o in run if o is not None]
        if not offs:
            i = j
            continue
        # run 可能跨越行内标签，拆成若干段连续偏移
        groups: list[list[int]] = [[offs[0]]]
        for o in offs[1:]:
            if o == groups[-1][-1] + 1:
                groups[-1].append(o)
            else:
                groups.append([o])
        for k, g in enumerate(groups):
            repl = keep if k == 0 else ""
            edits.append((g[0], g[-1] + 1, repl))
        i = j
    return edits


def insert_family(stack: str, family: str) -> str:
    """把 family 插到字体栈的**中文字体位置**：拉丁字体仍在它前面。

    落点依次是：第一个中文字体之前 → 末尾的 serif/sans-serif 之前 → 末尾。
    已经在栈里就原样返回（幂等）。
    """
    quoted = f'"{family}"'
    items = [s.strip() for s in stack.split(",")]
    if any(i.strip("\"'") == family for i in items):
        return stack
    at = next((k for k, i in enumerate(items) if _CJK_FAMILY_RE.search(i)), None)
    if at is None:
        at = next((k for k, i in enumerate(items)
                   if i.strip("\"'").lower() in _GENERIC_FAMILIES), len(items))
    return ", ".join(items[:at] + [quoted] + items[at:])


def build_style(font_stack: str, line_height: str, mono_stack: str,
                text_align: str, letter_spacing: str, font_face: str = "") -> str:
    return (
        f'<style id="{STYLE_ID}" type="text/css">\n'
        f"/* injected by polish_cjk.py — 中文译文块的排版；重复运行会整段替换 */\n"
        f"{font_face}"
        f".{CLASS_NAME}, *[lang=\"zh-CN\"], *[lang=\"zh\"], *[lang=\"zh-Hans\"] {{\n"
        f"  font-family: {font_stack} !important;\n"
        f"  line-height: {line_height} !important;\n"
        f"  text-align: {text_align};\n"
        f"  text-justify: inter-ideograph;\n"
        f"  letter-spacing: {letter_spacing};\n"
        f"  word-break: normal;\n"
        f"  overflow-wrap: break-word;\n"
        f"  -webkit-hyphens: none;\n"
        f"  hyphens: none;\n"
        f"}}\n"
        f".{CLASS_NAME} code, .{CLASS_NAME} kbd, .{CLASS_NAME} samp,\n"
        f".{CLASS_NAME} tt, .{CLASS_NAME} pre {{\n"
        f"  font-family: {mono_stack} !important;\n"
        f"  line-height: normal;\n"
        f"  letter-spacing: normal;\n"
        f"}}\n"
        f"/* 颜色一律不设：深色模式由原书的 @media (prefers-color-scheme:dark)\n"
        f"   规则继续管，译文块跟原文块同色。 */\n"
        f"</style>"
    )


def process_document(text: str, style: str, threshold: float, min_cjk: int,
                     collect_stats: bool = False) -> tuple[str, dict]:
    tree = _Tree(text)
    edits: list[tuple[int, int, str]] = []
    stats = {"cjk_blocks": 0, "p_blocks": 0, "lead_ws": 0, "trail_ws": 0,
             "inner_ws": 0, "mid_ws": 0, "marked": 0}

    for node in _walk(tree.root):
        if node.tag not in BLOCK_TAGS or node.tag in OPAQUE_TAGS:
            continue
        if node.end_start is None:          # 没闭合，跳过
            continue
        # 块级后代在 _stream 里被折成哨兵，所以这里判定的是"本层自己的文字"：
        # 容器 <div>（本层只有空白）永远不会被打标，而正文里嵌了一段
        # <pre> 的译文段落仍然能被正确认出来。
        stream = _stream(node, text)
        if not _is_cjk_block(stream, threshold, min_cjk):
            continue

        stats["cjk_blocks"] += 1
        if node.tag == "p":
            stats["p_blocks"] += 1
        if stream and stream[0][0] in _WS:
            stats["lead_ws"] += 1
        if stream and stream[-1][0] in _WS:
            stats["trail_ws"] += 1

        raw = text[node.raw_start : node.raw_end]
        new_tag = _mark_starttag(raw)
        if new_tag is not None:
            edits.append((node.raw_start, node.raw_end, new_tag))
            stats["marked"] += 1

        for s, e, repl in _space_edits(stream):
            if text[s:e] != repl:
                edits.append((s, e, repl))
                stats["mid_ws"] += 1

    # <head> 里注入样式（幂等：已有同 id 的 <style> 整段替换）
    head = next((n for n in _walk(tree.root) if n.tag == "head"), None)
    if head is not None and head.end_start is not None:
        existing = next(
            (n for n in _walk(head)
             if n.tag == "style" and f'id="{STYLE_ID}"' in text[n.raw_start : n.raw_end]),
            None,
        )
        if existing is not None and existing.end_end is not None:
            edits.append((existing.raw_start, existing.end_end, style))
        else:
            edits.append((head.end_start, head.end_start, "\n" + style + "\n"))

    for s, e, repl in sorted(edits, key=lambda x: x[0], reverse=True):
        text = text[:s] + repl + text[e:]
    return text, stats


# ── 内嵌字体 ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FontPlan:
    """一次内嵌所需的全部东西：放哪、叫什么、内容是什么。"""

    zip_path: str        # zip 内的完整路径，如 "OEBPS/Misc/EpubTranslatorZhSans.woff"
    href: str            # OPF manifest 里的 href（相对 OPF 所在目录）
    media_type: str      # "font/woff"
    css_format: str      # @font-face 里 format() 的关键字
    family: str
    data: bytes
    chars: int           # 子集实际收录的字符数


def gb2312_charset() -> set[str]:
    """GB2312-80 全量（6763 汉字 + 682 符号 = 7445 个），程序化生成。"""
    out: set[str] = set()
    for hi in range(0xA1, 0xF8):
        for lo in range(0xA1, 0xFF):
            try:
                out.add(bytes((hi, lo)).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return out


_NON_CONTENT_RE = re.compile(r"<(style|script)\b.*?</\1\s*>", re.S | re.I)


def collect_cjk_chars(texts) -> set[str]:
    """全书**会被渲染的**中日韩字符（含全角标点）。

    两处讲究：
    - 先剥掉 <style>/<script>，它们不是正文。这里注入的字体栈本身就含
      "微软雅黑" 三个字，不剥的话第二次运行会把它们并进子集、字体字节
      变多，幂等性就没了。
    - 再 unescape 一遍，`&#x4e2d;` 这种数字实体写法的汉字也要算进去，
      否则子集会缺字。
    """
    out: set[str] = set()
    for text in texts:
        stripped = _NON_CONTENT_RE.sub(" ", text)
        out.update(ch for ch in html.unescape(stripped) if is_cjk_wide(ch))
    return out


def subset_font(src: Path, font_number: int, chars: set[str], flavor: str) -> bytes:
    """用 fontTools 现做一个只含 chars 的子集，返回文件字节。

    源字体缺失/装不上就抛 RuntimeError——绝不静默返回空，那会让人以为
    字体嵌上了。
    """
    try:
        from fontTools import subset as ft_subset
        from fontTools.ttLib import TTFont
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise RuntimeError(
            f"内嵌字体需要 fontTools，但导入失败：{exc}\n"
            f"  装它：uv add fonttools（或 uv run python -m pip install fonttools）\n"
            f"  不想内嵌：加 --no-embed-font"
        ) from exc

    if not src.exists():
        raise RuntimeError(
            f"字体源不存在：{src}\n"
            f"  Debian/Ubuntu 上装它：sudo apt install fonts-noto-cjk\n"
            f"  或用 --font-file 指向别的中文字体（.ttf/.otf/.ttc，"
            f".ttc 再配 --font-number）\n"
            f"  不想内嵌：加 --no-embed-font"
        )

    try:
        # recalcTimestamp=False：否则每次 save 都会写入当前时间，
        # 同样的输入会产出不同的字节，幂等性就没了。
        font = TTFont(src, fontNumber=font_number, recalcTimestamp=False,
                      lazy=False)
    except Exception as exc:
        raise RuntimeError(
            f"打不开字体源 {src}（--font-number {font_number}）：{exc}\n"
            f"  .ttc 集合里每个 index 是一个字体，可能是 index 越界；\n"
            f"  不想内嵌：加 --no-embed-font"
        ) from exc

    options = ft_subset.Options()
    options.layout_features = []       # 中文正文不需要连字/花体等 OpenType 特性
    options.hinting = False
    options.desubroutinize = True      # 展开 CFF 子程序，WOFF 压得更小
    options.name_IDs = ["*"]
    options.name_legacy = True
    options.notdef_outline = False
    options.drop_tables += ["DSIG"]
    options.recalc_bounds = False
    options.recalc_timestamp = False

    subsetter = ft_subset.Subsetter(options=options)
    subsetter.populate(unicodes=sorted(ord(c) for c in chars))
    subsetter.subset(font)

    font.flavor = None if flavor in ("opentype", "truetype") else flavor
    buf = io.BytesIO()
    font.save(buf)
    font.close()
    return buf.getvalue()


def _font_dir(names, opf_dir: str) -> str:
    """字体放哪：跟着本书已有字体的目录走，没有就用 Misc/。"""
    prefix = f"{opf_dir}/" if opf_dir else ""
    dirs = collections.Counter()
    for n in names:
        if not n.lower().endswith((".woff", ".woff2", ".otf", ".ttf", ".ttc")):
            continue
        if prefix and not n.startswith(prefix):
            continue
        dirs[posixpath.dirname(n[len(prefix):])] += 1
    if not dirs:
        return FONT_DIR_FALLBACK
    # 最常见的那个目录；同票按名字定序，保证结果确定。
    return min(dirs.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def plan_font(names, opf_dir: str, *, font_file: Path, font_number: int,
              coverage: str, book_chars: set[str], suffix: str = ".woff") -> FontPlan:
    chars = set(book_chars)
    if coverage == "gb2312":
        chars |= gb2312_charset()
    if not chars:
        raise RuntimeError("全书没有中日韩字符，没有可内嵌的字体子集"
                           "（不该内嵌的话加 --no-embed-font）")
    media_type, css_format = FONT_MEDIA_TYPES[suffix]
    data = subset_font(font_file, font_number, chars, css_format)

    rel_dir = _font_dir(names, opf_dir)
    href = posixpath.join(rel_dir, EMBED_FILE_STEM + suffix) if rel_dir \
        else EMBED_FILE_STEM + suffix
    zip_path = posixpath.join(opf_dir, href) if opf_dir else href
    return FontPlan(zip_path=zip_path, href=href, media_type=media_type,
                    css_format=css_format, family=EMBED_FAMILY, data=data,
                    chars=len(chars))


def font_face_css(plan: FontPlan, doc_name: str) -> str:
    """一段 @font-face。

    src 是**相对于这份文档**的路径——样式是内联在每个 XHTML 的 <head> 里的，
    不是外部 CSS，所以基准目录是文档自己的目录（同一本书里文档可能分处
    不同深度，必须逐个文档算）。
    """
    doc_dir = posixpath.dirname(doc_name)
    rel = posixpath.relpath(plan.zip_path, doc_dir or ".")
    return (
        f"@font-face {{\n"
        f'  font-family: "{plan.family}";\n'
        f"  font-style: normal;\n"
        f"  font-weight: normal;\n"
        f"  font-display: swap;\n"
        f'  src: url({rel}) format("{plan.css_format}");\n'
        f"}}\n"
    )


def register_font_in_opf(opf_text: str, plan: FontPlan) -> str:
    """在 manifest 里登记字体。已登记过就原地替换那一条（幂等）。"""
    item = (f'<item href="{plan.href}" id="{FONT_ITEM_ID}" '
            f'media-type="{plan.media_type}" />')
    existing = re.search(
        r'[ \t]*<item\b[^>]*\b(?:id="' + re.escape(FONT_ITEM_ID)
        + r'"|href="' + re.escape(plan.href) + r'")[^>]*/>[ \t]*\n?',
        opf_text,
    )
    if existing:
        indent = re.match(r"[ \t]*", existing.group(0)).group(0)
        tail = "\n" if existing.group(0).endswith("\n") else ""
        return opf_text[:existing.start()] + indent + item + tail \
            + opf_text[existing.end():]
    close = re.search(r"([ \t]*)</manifest>", opf_text)
    if not close:
        raise RuntimeError("OPF 里找不到 </manifest>，无法登记字体")
    indent = close.group(1) + "  "
    return (opf_text[:close.start()] + indent + item + "\n"
            + opf_text[close.start():])


# ── EPUB 层 ─────────────────────────────────────────────────────────────

_XHTML_SUFFIXES = (".xhtml", ".html", ".htm")


def _is_doc(name: str) -> bool:
    return name.lower().endswith(_XHTML_SUFFIXES)


def find_opf(names, container: bytes | None) -> str:
    """OPF 的 zip 内路径：先信 container.xml，退化到第一个 .opf。"""
    if container:
        m = re.search(rb'full-path="([^"]+)"', container)
        if m:
            return m.group(1).decode("utf-8")
    opfs = sorted(n for n in names if n.lower().endswith(".opf"))
    if not opfs:
        raise RuntimeError("EPUB 里找不到 OPF（META-INF/container.xml 也没有指）")
    return opfs[0]


def polish_epub(src: Path, dst: Path, *, style_builder, threshold: float = 0.15,
                min_cjk: int = 1, stats_only: bool = False,
                embed_font: bool = True, font_file: Path = DEFAULT_FONT_FILE,
                font_number: int = DEFAULT_FONT_NUMBER,
                font_coverage: str = "book") -> dict:
    """style_builder(font_face_css) -> <style> 整段；每个文档单独调一次，
    因为 @font-face 的 src 要按该文档所在目录算相对路径。"""
    total = {"docs": 0, "cjk_blocks": 0, "p_blocks": 0, "lead_ws": 0,
             "trail_ws": 0, "mid_ws": 0, "marked": 0,
             "font": None, "font_bytes": 0, "font_chars": 0, "book_chars": 0}
    with zipfile.ZipFile(src) as zin:
        infos = zin.infolist()
        raw = {info.filename: zin.read(info) for info in infos}
        docs = [n for n in raw if _is_doc(n)]
        texts = {n: raw[n].decode("utf-8") for n in docs}

        book_chars = collect_cjk_chars(texts.values())
        total["book_chars"] = len(book_chars)

        plan = None
        if embed_font and not stats_only:
            opf_name = find_opf(raw, raw.get("META-INF/container.xml"))
            plan = plan_font(list(raw), posixpath.dirname(opf_name),
                             font_file=font_file, font_number=font_number,
                             coverage=font_coverage, book_chars=book_chars)
            total["font"] = plan.zip_path
            total["font_bytes"] = len(plan.data)
            total["font_chars"] = plan.chars

        payload = dict(raw)
        for name in docs:
            face = font_face_css(plan, name) if plan else ""
            new_text, st = process_document(texts[name], style_builder(face),
                                            threshold, min_cjk)
            total["docs"] += 1
            for k in ("cjk_blocks", "p_blocks", "lead_ws", "trail_ws", "mid_ws", "marked"):
                total[k] += st[k]
            payload[name] = new_text.encode("utf-8")

        if stats_only:
            return total

        ordered = list(infos)
        if plan is not None:
            payload[plan.zip_path] = plan.data
            payload[opf_name] = register_font_in_opf(
                raw[opf_name].decode("utf-8"), plan).encode("utf-8")
            if plan.zip_path not in raw:
                # 上一轮已经嵌过就落在原位（连跑两次输出逐字节相同）；
                # 第一次才追加，用固定时间戳保持确定性。
                zi = zipfile.ZipInfo(plan.zip_path, date_time=FONT_EPOCH)
                zi.external_attr = 0o644 << 16
                ordered.append(zi)

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        # mimetype 必须是第一个条目且不压缩；其余条目保持源 zip 的顺序、
        # 时间戳与属性，未改动的文件因此逐字节不变，输出也是确定性的。
        ordered.sort(key=lambda i: i.filename != "mimetype")
        with zipfile.ZipFile(dst, "w") as zout:
            for info in ordered:
                zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                zi.external_attr = info.external_attr
                zi.internal_attr = info.internal_attr
                zi.create_system = info.create_system
                zi.comment = info.comment
                ct = (zipfile.ZIP_STORED if info.filename == "mimetype"
                      else zipfile.ZIP_DEFLATED)
                zout.writestr(zi, payload[info.filename], compress_type=ct)
    return total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="输入 EPUB")
    ap.add_argument("dst", type=Path, nargs="?", help="输出 EPUB（--stats 时可省略）")
    ap.add_argument("--stats", action="store_true", help="只统计，不写文件")
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="块内 2*CJK/(2*CJK+拉丁字母) 的阈值（默认 0.15）")
    ap.add_argument("--min-cjk", type=int, default=1, help="块内最少 CJK 字符数（默认 1）")
    ap.add_argument("--line-height", default="1.75")
    ap.add_argument("--font-family", default=DEFAULT_FONT_STACK)
    ap.add_argument("--mono-family", default=DEFAULT_MONO_STACK)
    ap.add_argument("--text-align", default="justify")
    ap.add_argument("--letter-spacing", default="0.01em")
    ap.add_argument("--embed-font", dest="embed_font", action="store_true",
                    default=True,
                    help="内嵌中文字体子集（默认开）")
    ap.add_argument("--no-embed-font", dest="embed_font", action="store_false",
                    help="不内嵌字体，只靠系统字体栈（省体积）")
    ap.add_argument("--font-coverage", choices=("book", "gb2312"), default="book",
                    help="子集范围：book=只含本书用字（默认，约 200 KB）；"
                         "gb2312=再并上 GB2312 全量 7445 字（约 1.2 MB，"
                         "读者自己加批注也不掉字体）")
    ap.add_argument("--font-file", type=Path, default=DEFAULT_FONT_FILE,
                    help=f"字体源（默认 {DEFAULT_FONT_FILE}）")
    ap.add_argument("--font-number", type=int, default=DEFAULT_FONT_NUMBER,
                    help=f".ttc 集合里的索引（默认 {DEFAULT_FONT_NUMBER}"
                         f" = Noto Sans CJK SC）")
    args = ap.parse_args(argv)

    if not args.stats and args.dst is None:
        ap.error("需要输出路径（或用 --stats）")

    font_stack = args.font_family
    if args.embed_font and not args.stats:
        font_stack = insert_family(font_stack, EMBED_FAMILY)

    def style_builder(font_face: str) -> str:
        return build_style(font_stack, args.line_height, args.mono_family,
                           args.text_align, args.letter_spacing, font_face)

    try:
        total = polish_epub(args.src, args.dst or args.src,
                            style_builder=style_builder,
                            threshold=args.threshold, min_cjk=args.min_cjk,
                            stats_only=args.stats, embed_font=args.embed_font,
                            font_file=args.font_file,
                            font_number=args.font_number,
                            font_coverage=args.font_coverage)
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print(f"文档 {total['docs']} 个 / 中文块 {total['cjk_blocks']} 个"
          f"（其中 <p> {total['p_blocks']} 个）")
    print(f"  以空白开头 {total['lead_ws']} / 以空白结尾 {total['trail_ws']}")
    if not args.stats:
        print(f"  打标 {total['marked']} 个块，清理空格 {total['mid_ws']} 处")
        if total["font"]:
            print(f"  内嵌字体 {total['font']}："
                  f"{total['font_chars']} 字 / {total['font_bytes'] / 1024:.0f} KB"
                  f"（全书用字 {total['book_chars']} 个）")
        else:
            print(f"  未内嵌字体（全书用字 {total['book_chars']} 个）")
        print(f"  -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
