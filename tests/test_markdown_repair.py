"""Tests for the markdown the OCR engine gets wrong, step 2 of pdf_to_epub.

Every input here is text PaddleOCR-VL 1.6 actually produced, taken from arXiv
2608.25512 ("Spatiotemporal Composability", 92 A4 pages), which is the paper
each of these three faults was found on.
"""

import pdf_to_epub as P

# The OCR of a commutative diagram: \begin{array} came out unbalanced and the
# closing \] was never written, so pandoc read on into the next paragraph.
UNCLOSED = (
    r"\[\begin{array}{c}\Gamma\xrightarrow{f_{1}}\Gamma\xrightarrow{\quad\quad}"
    r"\Gamma\\\partial\Gamma\xrightarrow{f_{1}^{\prime}}\partial\Gamma\\\quad"
)

PROSE = (
    "The witness holds each returned inverse to one equation,  $g(\\delta) = "
    "\\gamma$: the inverse is required to revert the effect only at the state "
    "where it was applied."
)

ALGORITHM_1 = """Algorithm 1 Effect tracking

1 async function execute(callback, guard)
2 iter ← callback()
3 inverse ← id
4 while guard()
5 | (value, done) ← await iter.next()
6 if value then inverse ← value ○ inverse
7 if done then break
8 return inverse
9 function effect(ctx, callback)
10 armed ← true
11 task ← execute(callback, () → armed)
12 async function dispose()
13 if not armed then return
14 armed ← false
15 recover ← await task
16 recover()
17 ctx.dispose ← dispose ○ ctx.dispose
18 return dispose
"""

# Algorithm 6: no step numbers, and the engine dropped the indentation.
ALGORITHM_6 = """Algorithm 6 Proxy-mediated context access

function resolve(ctx, key)
fiber ← ctx.fiber
repeat
if key ∈ fiber.committed then return fiber.committed[key]
if key ∈ fiber.inject then throw INACTIVE_ACCESS
if fiber = root then throw UNDECLARED_ACCESS
fiber ← fiber.parent.fiber
"""

# Algorithm 8 fell across a page break, so its steps arrive as two paragraphs.
ALGORITHM_8 = """Algorithm 8 Module classification

1 function classify(stashed, externals)
2 accepted ← ∅
3 declined ← externals
4 pending ← stashed
5 for url in stashed do

6 | pending ← pending ∪ get_imports(url)
7 | if url ∈ declined then continue
8 | accepted ← accepted ∪ {url}
9 | return (accepted, declined)
"""


def blocks(text: str) -> list[list[str]]:
    """The fenced blocks of `text`, as lists of their lines."""
    out, current = [], None
    for line in text.split("\n"):
        if line.strip() == "```":
            if current is None:
                current = []
            else:
                out.append(current)
                current = None
        elif current is not None:
            current.append(line)
    assert current is None, "unbalanced fence"
    return out


class TestCloseDisplayMath:
    def test_closes_the_opener_at_the_end_of_its_paragraph(self):
        text, brackets, dollars = P._close_display_math(
            f"{UNCLOSED}\n\n{PROSE}\n\nnext paragraph\n"
        )
        assert (brackets, dollars) == (1, 0)
        first = text.split("\n\n")[0]
        assert first.endswith("\\]")
        # The prose after the blank line is outside the formula, which is the
        # whole point: 6725 characters of it went into one span without this.
        assert PROSE in text.split("\\]")[1]

    def test_leaves_a_formula_that_runs_over_several_lines(self):
        text = "\\[\n\\begin{array}{c}\\Gamma\\\\\\partial\\Gamma\n\\end{array}\n\\]\n"
        assert P._close_display_math(text) == (text, 0, 0)

    def test_closes_an_odd_display_dollar_pair(self):
        text, brackets, dollars = P._close_display_math("$$ \\gamma^{\\prime}\n\nafter\n")
        assert (brackets, dollars) == (0, 1)
        assert text.split("\n")[0].endswith("$$")

    def test_leaves_a_balanced_paragraph_byte_identical(self):
        text = f"{PROSE}\n\n$$ x = y $$\n\nmore prose\n"
        assert P._close_display_math(text) == (text, 0, 0)

    def test_ignores_fenced_code(self):
        text = "```\n\\[ unclosed in code\n```\n"
        assert P._close_display_math(text) == (text, 0, 0)


class TestTightenInlineMath:
    def test_tightens_a_padded_pair(self):
        # pandoc's tex_math_dollars rejects this, and markdown+raw_tex then eats
        # the \Gamma, so the reader gets "$ _n $".
        text, pairs = P._tighten_inline_math("the context  $ \\Gamma_n $  is fresh\n")
        assert pairs == 1
        assert text == "the context  $\\Gamma_n$  is fresh\n"

    def test_keeps_the_pairs_paired(self):
        # A tight pair must still be matched, or the gap after it — here
        # "$ yields a pair $" — is read as the next formula and the spaces
        # around the real formulas are eaten.
        text, pairs = P._tighten_inline_math(
            "where  $e(\\gamma)$ yields a pair  $(\\delta, g)$ representing:\n"
        )
        assert pairs == 0
        assert text == "where  $e(\\gamma)$ yields a pair  $(\\delta, g)$ representing:\n"

    def test_leaves_an_empty_pair_alone(self):
        assert P._tighten_inline_math("a $ $ b\n") == ("a $ $ b\n", 0)

    def test_leaves_display_math_alone(self):
        text = "$$ \\gamma^{\\prime}[\\delta_{k}]=d_{1} $$\n"
        assert P._tighten_inline_math(text) == (text, 0)

    def test_leaves_a_multiline_display_block_alone(self):
        text = "$$\n x = $ y $\n$$\nand  $ z $  after\n"
        out, pairs = P._tighten_inline_math(text)
        assert pairs == 1
        assert out == "$$\n x = $ y $\n$$\nand  $z$  after\n"

    def test_leaves_fenced_code_alone(self):
        text = "```\nprintf(\"$ %d $\", n)\n```\n"
        assert P._tighten_inline_math(text) == (text, 0)


class TestFencePseudocode:
    def test_fences_a_captioned_listing_and_keeps_every_line(self):
        text, fenced = P._fence_pseudocode(ALGORITHM_1)
        assert fenced == 1
        body = blocks(text)[0]
        assert len(body) == 18
        assert body[0] == "1 async function execute(callback, guard)"
        assert body[-1] == "18 return dispose"
        assert "Algorithm 1 Effect tracking" not in "\n".join(body)

    def test_fences_a_listing_with_no_step_numbers(self):
        text, fenced = P._fence_pseudocode(ALGORITHM_6)
        assert fenced == 1
        body = blocks(text)[0]
        assert len(body) == 7
        assert body[-1] == "fiber ← fiber.parent.fiber"

    def test_rejoins_a_listing_split_across_a_page_break(self):
        text, fenced = P._fence_pseudocode(ALGORITHM_8)
        assert fenced == 1
        body = blocks(text)[0]
        assert len(body) == 9
        assert "" not in body

    def test_fences_an_uncaptioned_listing(self):
        text, fenced = P._fence_pseudocode(
            ALGORITHM_1.split("\n", 2)[2]  # drop the caption the engine may miss
        )
        assert fenced == 1
        assert len(blocks(text)[0]) == 18

    def test_leaves_prose_alone(self):
        # Real paragraphs from the paper, including one that opens with a word
        # the step pattern would match if it were lowercase.
        prose = (
            f"{PROSE}\n\n"
            "Algorithm 1 shows the construction of ctx. effect. We write  "
            "$f \\circ g$  for the disposer that runs f after g, and id for the "
            "no-op; prepending each new inverse therefore yields LIFO recovery.\n\n"
            "Entries. A configuration consists of entries. Each entry specifies "
            "a fiber and manages it.\n"
        )
        assert P._fence_pseudocode(prose) == (prose, 0)

    def test_leaves_a_short_run_of_prose_lines_alone(self):
        # The definition lists the engine emits one short line per item: four
        # consecutive short lines, and not one of them code.
        text = (
            "id — a stable identifier, used as the reconciliation key;\n"
            "url — the URL of the component module to instantiate;\n"
            "isolate — an isolation annotation applied to the context;\n"
            "disabled — whether the entry is administratively turned off.\n"
        )
        assert P._fence_pseudocode(text) == (text, 0)

    def test_leaves_a_table_alone(self):
        text = (
            "| page set | wall time | peak VRAM |\n"
            "| --- | --- | --- |\n"
            "| 3 two-column pages | 46 s | 9876 MiB |\n"
            "| 1 page, 3x3 matrix | 40 s | 8889 MiB |\n"
        )
        assert P._fence_pseudocode(text) == (text, 0)

    def test_leaves_an_existing_fence_alone(self):
        text, fenced = P._fence_pseudocode("```\n" + ALGORITHM_1.split("\n", 2)[2] + "```\n")
        assert fenced == 0
        assert text.count("```") == 2
