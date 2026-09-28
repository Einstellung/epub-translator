"""Tests for fix_layout.py, the post-pass over append-block bilingual EPUBs.

The fixtures mimic what epub_translator writes: the translation is a clone of
the original block (same tag, same attributes, id suffixed `__translated`)
inserted directly after it.
"""

import zipfile

from lxml import etree

import fix_layout as F
import polish_cjk

HEAD = '<?xml version="1.0" encoding="utf-8"?>\n'


def doc(body: str, html_attrs: str = ' xml:lang="en"') -> str:
    return (
        HEAD
        + f'<html xmlns="http://www.w3.org/1999/xhtml" '
        f'xmlns:epub="http://www.idpf.org/2007/ops"{html_attrs}>'
        f"<head><title>t</title></head><body>{body}</body></html>"
    )


def fix(body: str, html_attrs: str = ' xml:lang="en"') -> str:
    out, _ = F.process_document(doc(body, html_attrs))
    etree.fromstring(out.encode())  # still well-formed XHTML
    return out[out.index("<body>") + 6 : out.index("</body>")]


# ── 1. lang ─────────────────────────────────────────────────────────────


def test_translation_block_gets_zh_lang():
    body = '<p class="x">A plain sentence.</p><p class="x">一个普通的句子。</p>'
    assert fix(body) == (
        '<p class="x">A plain sentence.</p>'
        '<p class="x" lang="zh" xml:lang="zh">一个普通的句子。</p>'
    )


def test_copied_source_lang_is_replaced():
    body = '<p lang="en" xml:lang="en">Some text.</p><p lang="en" xml:lang="en">一些文字。</p>'
    assert fix(body) == (
        '<p lang="en" xml:lang="en">Some text.</p>'
        '<p lang="zh" xml:lang="zh">一些文字。</p>'
    )


def test_no_xml_lang_when_the_document_uses_none():
    body = "<h2>Getting started</h2><h2>入门</h2>"
    assert fix(body, html_attrs="") == '<h2>Getting started</h2><h2 lang="zh">入门</h2>'


def test_block_already_tagged_zh_is_left_alone():
    body = (
        '<p class="x">Some text.</p>'
        '<p class="x zh-translation" lang="zh-CN" xml:lang="zh-CN">一些文字。</p>'
    )
    assert fix(body) == body


def test_mixed_text_and_list_item_is_left_alone():
    # text beside a child list is translated in append-text form: no clone
    body = (
        '<ul><li class="b" id="p1">Training has stages: 训练分几个阶段：'
        "<ul><li>Pretraining</li><li>预训练</li></ul></li></ul>"
    )
    out = fix(body)
    assert out.startswith('<ul><li class="b" id="p1">Training has stages: 训练分几个阶段：<ul>')
    assert '<li>Pretraining<div lang="zh" xml:lang="zh">预训练</div></li>' in out


def test_different_attributes_are_not_a_pair():
    body = '<p class="a">Some text.</p><p class="b">一些文字。</p>'
    assert fix(body) == body


# ── 2. list items ───────────────────────────────────────────────────────


def test_list_item_translation_moves_inside_the_item():
    body = (
        "<ol>\n<li>Install the tool.</li><li> 安装工具。 </li>\n"
        "<li>Run it.</li><li> 运行它。 </li>\n</ol>"
    )
    out = fix(body)
    assert out == (
        '<ol>\n<li>Install the tool.<div lang="zh" xml:lang="zh"> 安装工具。 </div></li>\n'
        '<li>Run it.<div lang="zh" xml:lang="zh"> 运行它。 </div></li>\n</ol>'
    )
    assert out.count("<li") == 2  # numbering unchanged


def test_polished_list_item_keeps_its_class_lang_and_id():
    body = (
        '<ul><li class="item" id="p1">Check the blog.</li>'
        '<li class="item zh-translation" id="p1__translated" lang="zh-CN" '
        'xml:lang="zh-CN">查看博客。</li></ul>'
    )
    assert fix(body) == (
        '<ul><li class="item" id="p1">Check the blog.'
        '<div lang="zh-CN" xml:lang="zh-CN" class="zh-translation" id="p1__translated">'
        "查看博客。</div></li></ul>"
    )


def test_list_item_with_block_children_is_left_alone():
    # the translator already put the translation inside: <li><p>en</p><p>zh</p></li>
    body = "<ul><li><p>Fill in:</p><p>填写：</p></li><li><p>Next step.</p><p>下一步。</p></li></ul>"
    out = fix(body)
    assert out.count("<li>") == 2
    assert '<p lang="zh" xml:lang="zh">填写：</p>' in out


def test_toc_items_with_mixed_text_are_not_merged():
    # append-text TOC entries: both items carry CJK, neither is a clone
    body = (
        '<ol><li data-type="s"><a href="a.html">The Block 块</a></li>'
        '<li data-type="s"><a href="b.html">Positional Embeddings 位置嵌入</a></li></ol>'
    )
    assert fix(body) == body


# ── 4. notes ────────────────────────────────────────────────────────────


def test_footnote_aside_translation_moves_into_the_note():
    body = (
        '<aside id="fn1" epub:type="footnote">A note in English.</aside>'
        '<aside id="fn1__translated" epub:type="footnote"> 英文注释。 </aside>'
    )
    assert fix(body) == (
        '<aside id="fn1" epub:type="footnote">A note in English.'
        '<div lang="zh" xml:lang="zh" id="fn1__translated"> 英文注释。 </div></aside>'
    )


def test_merged_note_keeps_one_backlink():
    body = (
        '<aside id="n" type="footnote"><a href="ch1.html#r">1</a> See appendix '
        '<a href="app.html#a">A</a>.</aside>'
        '<aside id="n__translated" type="footnote"><a href="ch1.html#r">1</a> 见附录 '
        '<a href="app.html#a">A</a>。</aside>'
    )
    out = fix(body)
    assert out.count('href="ch1.html#r"') == 1
    assert out.count('href="app.html#a"') == 2  # ordinary links stay
    assert out.count("<aside") == 1


def test_backlink_wrapper_is_dropped_from_translated_paragraph_in_note():
    body = (
        '<div class="footnote" role="doc-footnote">'
        '<p class="fn"><span class="num"><a href="c.html#ref" role="doc-backlink">*</a> </span>'
        "All references are at the back.</p>"
        '<p class="fn"><span class="num"><a href="c.html#ref" role="doc-backlink">＊</a></span>'
        " 所有参考文献都在书后。</p></div>"
    )
    out = fix(body)
    assert out.count('href="c.html#ref"') == 1
    assert '<p class="fn" lang="zh" xml:lang="zh"> 所有参考文献都在书后。</p>' in out


def test_leading_link_outside_a_note_is_kept():
    body = (
        '<p class="x"><a href="c2.html">Skip ahead</a> to chapter two.</p>'
        '<p class="x"><a href="c2.html">跳到</a>第二章。</p>'
    )
    assert fix(body).count('href="c2.html"') == 2


# ── 6. TOC numbering ────────────────────────────────────────────────────


def nav(entries: list[str]) -> str:
    items = "".join(f'<li><a href="c.xhtml">{e}</a></li>' for e in entries)
    return f'<nav epub:type="toc"><ol>{items}</ol></nav>'


def test_nav_drops_the_repeated_section_number():
    out = fix(nav([
        "I. INTRODUCTION I. 引言",
        "1 Getting Started 1 入门",
        "2.3. Model and Representation 2.3. 模型与表征",
        "A. Related Works A. 相关工作",
    ]))
    for want in ("I. INTRODUCTION 引言", "1 Getting Started 入门",
                 "2.3. Model and Representation 模型与表征", "A. Related Works 相关工作"):
        assert f">{want}<" in out


def test_nav_repeat_inside_the_english_title_is_kept():
    out = fix(nav(["1 Top 1 Results 1 结果"]))
    assert ">1 Top 1 Results 结果<" in out


def test_nav_entries_without_a_repeated_number_are_untouched():
    entries = ["Cover 封面", "I Robot 我，机器人", "Chapter 1 Basics 第1章 基础", "Preface"]
    body = nav(entries)
    assert fix(body) == body


def test_links_outside_nav_are_untouched():
    body = '<p><a href="c.xhtml">1 Intro 1 引言</a></p>'
    assert fix(body) == body


def test_ncx_label_drops_the_repeated_number():
    ncx = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><navMap>'
        '<navPoint id="p1"><navLabel><text>1.1. Model 1.1. 模型</text></navLabel>'
        '<content src="c.html"/></navPoint></navMap></ncx>'
    )
    out, stats = F.process_document(ncx)
    assert "<text>1.1. Model 模型</text>" in out
    assert stats["toc numbers deduplicated"] == 1


# ── whole-document properties ───────────────────────────────────────────


def test_idempotent():
    body = (
        '<ul><li>One.</li><li>一。</li></ul><p>Two.</p><p>二。</p>'
        '<aside id="f" epub:type="footnote"><a href="#r">1</a> Note.</aside>'
        '<aside id="f__translated" epub:type="footnote"><a href="#r">1</a> 注。</aside>'
        + nav(["1 Intro 1 引言"])
    )
    once, _ = F.process_document(doc(body))
    twice, stats = F.process_document(once)
    assert twice == once
    assert sum(stats.values()) == 0


def test_polish_cjk_tags_the_moved_translation():
    fixed, _ = F.process_document(doc("<ul><li>One item.</li><li>一项。</li></ul>"))
    polished, _ = polish_cjk.process_document(fixed, "<style></style>", 0.15, 1)
    assert '<li>One item.<div lang="zh" xml:lang="zh" class="zh-translation">一项。</div></li>' in polished


def test_fix_epub_rewrites_only_what_changed(tmp_path):
    src = tmp_path / "in.epub"
    untouched = doc("<p>English only.</p>").encode()
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("OEBPS/a.xhtml", doc("<ol><li>One.</li><li>一。</li></ol>"),
                   compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/b.xhtml", untouched, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("OEBPS/img.png", b"\x89PNG", compress_type=zipfile.ZIP_DEFLATED)
    dst = tmp_path / "out.epub"
    stats = F.fix_epub(src, dst)
    assert stats["list items merged"] == 1
    assert stats["documents changed"] == 1
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(dst) as b:
        assert b.namelist() == a.namelist()
        assert b.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert b.read("OEBPS/b.xhtml") == untouched
        assert b.read("OEBPS/img.png") == a.read("OEBPS/img.png")
        assert b.read("OEBPS/a.xhtml").count(b"<li") == 1


def test_fix_epub_refuses_to_overwrite_its_input(tmp_path):
    src = tmp_path / "in.epub"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
    try:
        F.fix_epub(src, src)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
