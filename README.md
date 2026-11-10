# pdf_ocr: PDF to Markdown, JSON and CSV with MiniCPM-OCR or PaddleOCR-VL, with a Score for every Block

`pdf_ocr.py` converts a PDF into Markdown, a detailed JSON file and one CSV per table. It uses only
the local models in `models/`:

| Model | Role |
|---|---|
| `models/PP-DocLayoutV3` | the Layout model: finds the blocks on each page (text, titles, tables, figures, …) and their reading order |
| `models/MiniCPM-OCR-1.0` | the default Reader: reads each block, with tables as HTML, formulas as LaTeX, and charts or seals if you ask |
| `models/PaddleOCR-VL-1.6` | the other Reader, chosen with `--reader paddle` |
| `models/MiniCPM-OCR-Jev` | the Judge: rates every text, formula and table Reading from 0 to 1 (a Score), on by default; `--no-judge` turns it off |

It downloads nothing, and the model-hoster connectivity check is switched off.

It is still one command. MiniCPM needs torch and transformers, which cannot share a virtual environment
with paddlepaddle ([ADR-0001](docs/adr/0001-torch-models-in-a-separate-venv.md)). So the torch part lives in
a second uv project, `worker/`, which the main command starts as a background process for the length of
the run. You never start it yourself.

## Setup

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). `pyproject.toml` pins `paddlepaddle-gpu` 3.3.1
from the CUDA 13.0 Paddle index. It also pulls in `paddleocr[doc-parser]` and `opencc-python-reimplemented`,
which the Simplified to Traditional repair needs.

```bash
uv sync                      # the main environment: the Layout model, PaddleOCR and the paddle Reader
uv sync --project worker     # the worker environment: torch + transformers for MiniCPM (about 5.6 GB, CUDA 13.2 wheels)
```

The worker environment is created in `worker/.venv`. It holds MiniCPM-OCR-1.0 and the Judge, so it is needed
unless you run `--reader paddle --no-judge`. If it is missing, the tool stops at once and tells you to run the
second command. The torch wheel from PyPI is built for CUDA 13.2, so the GPU driver must support it. The
two environments are separate processes, so their CUDA libraries never meet.

If no CUDA GPU is found, the tool runs on the CPU, which works but is slow. (The MiniCPM worker needs a CUDA GPU.)

`--models-dir` has to hold only the folders that your options use: `PP-DocLayoutV3` always, plus
`MiniCPM-OCR-1.0` for the default Reader or `PaddleOCR-VL-1.6` for `--reader paddle`, plus `MiniCPM-OCR-Jev`
unless you pass `--no-judge`. A missing folder is reported by name before anything is loaded.

**The Judge also needs three things from the machine, one of them a one-time download.** (Skip all of this with
`--no-judge`.) The tool checks them before it loads any model, and names what is missing:

```bash
uv run --project worker playwright install chromium   # once: headless Chromium for the renderer, ~650 MB in ~/.cache/ms-playwright
```

- **Chromium**, installed by the command above. Set `PLAYWRIGHT_BROWSERS_PATH` if you keep it elsewhere. On a
  fresh Linux machine `playwright install --with-deps chromium` also installs Chromium's system libraries (it
  needs root).
- **`node`** (Node.js) on `PATH`. The Judge's renderer screens each Reading's LaTeX with KaTeX in Node before
  drawing it; a Reading KaTeX cannot parse gets the verdict `precheck_reject`.
- **Noto Sans CJK** installed in the system (`fonts-noto-cjk`, `noto-fonts-cjk` or `google-noto-sans-cjk-fonts`,
  depending on the distribution; `fc-list | grep "Noto Sans CJK"` must find it). The renderer draws Latin
  letters, digits and CJK text with it.
- The **bundled font** `worker/fonts/DroidSansFallbackFull.ttf` ships in the repository (Apache-2.0, with its
  licence files) and fills the CJK glyphs Noto lacks. If it is missing the tool says so; `git checkout` restores it.

Why the system's Noto Sans CJK is needed even though the font is bundled, and why the renderer runs under its
own fontconfig, is explained in "The Judge" below.

Memory on a 12 GB GPU, measured with `nvidia-smi` (RTX 4070 Ti, 12,282 MiB, with about 2.4 GB used by the desktop
and a game during the measurements; a four-page run of a stand-in PDF, see below):

| path | main process | worker | peak of the whole card |
|---|---|---|---|
| default: Layout model + MiniCPM + Jev | 874 MiB | 7,514 MiB | 10,963 MiB |
| `--reader paddle` (Judge on): Layout model + PaddleOCR-VL, Jev in the worker | 8,126 MiB at start-up, less later | 5,006 MiB | 11,393 MiB |

So the default options fit with all three models loaded at once. A very long table adds up to about 1.4 GB
while Jev scores it. `--reader paddle` alone holds about 8 GB, and PaddleOCR-VL plus Jev side by side do not fit
(about 12.5 GB before the desktop). To make them fit, the run releases PaddleOCR-VL's cached memory after
construction and after every page's read, and moves Jev's weights to the CPU while PaddleOCR-VL reads, and back
before scoring (about a second per page). If your card is shared with other heavy processes and the Judge still runs
out of memory, the page or Block that hit it is recorded (a failed page, or the verdict `forward_error` with the
reason) and the run ends with a line saying so and suggesting `--no-judge` or `--reader minicpm`.

## Usage

```bash
uv run pdf_ocr.py input.pdf                         # -> output/<stem>/
uv run pdf_ocr.py input.pdf -o out --pages 1-3,7,10-
uv run pdf_ocr.py --help
```

The output goes to `output/<pdf stem>/` unless you pass `-o`:

```
<stem>.md      Markdown: headings, paragraphs, HTML tables, LaTeX formulas, <!-- page N --> markers
<stem>.json    per-page blocks with label, text, bbox, polygon, reading order, raw layout detections
imgs/          figures, charts and seals cropped from the pages (only if there are any)
csv/           one CSV per valid table: pNNN_tableKK.csv (only if there are any)
```

If any page fails, the run still finishes. The failure is recorded in the JSON (`source.pages_failed`)
and in the Markdown, and the exit code is 1.

### Options

| Option | Default | What it does |
|---|---|---|
| `-o, --output-dir` | `output/<stem>` | where to write the results |
| `--models-dir` | `./models` | folder containing `PP-DocLayoutV3/` and the Reader's folder (`MiniCPM-OCR-1.0/`, or `PaddleOCR-VL-1.6/` with `--reader paddle`) |
| `--pages` | all | 1-based page selection, e.g. `1-3,7,10-` |
| `--device` | `gpu:0` if available | Paddle device, e.g. `cpu` |
| `--reader {minicpm,paddle}` | `minicpm` | the model that reads each block: MiniCPM-OCR-1.0 (through the torch worker) or PaddleOCR-VL-1.6. With `paddle` the Markdown, the CSVs and the blocks in the JSON are the same as before MiniCPM was added; the JSON only gains a `reader` field in `engine` and `settings` |
| `--dpi` | 216 | page render resolution |
| `--layout-threshold` | 0.2 | layout score threshold; lower finds more blocks but too low invents boxes |
| `--max-new-tokens` | 8192 | decoding limit per block; raise it for very large tables. With MiniCPM it is the limit for tables only (see below) |
| `--charts` | off | turn charts into data tables (see Caveats) |
| `--seals` | off | read the text on seals and stamps |
| `--ocr-images` | off | also OCR any text inside figures |
| `--no-merge-blocks` | | do not merge adjacent text blocks before OCR |
| `--no-judge` | | do not score blocks (no Judge, no `playwright`, `node` or Noto Sans CJK needed). The ink-based re-read of short blocks runs instead |
| `--judge-threshold FLOAT` | 0.98 | blocks whose Score is below this are flagged `below_threshold` in the JSON and counted per page on stderr (see "The Judge" for what to expect at 0.98) |
| `--no-recover` | | with `--no-judge`, turn off the truncated-block re-read. With the Judge on that re-read never runs |
| `--script {auto,off}` | `auto` | Simplified to Traditional repair; `off` keeps the raw model output |
| `--md-ignore-labels` | header, footer, page number, aside | block labels left out of the Markdown (the JSON keeps everything) |
| `--no-merge-tables` | | keep a table that continues over a page break as separate per-page tables in the Markdown and CSV |
| `--no-csv` | | do not write `csv/` |
| `--no-page-markers` | | leave out the `<!-- page N -->` markers |
| `-v, --verbose` | | show the PaddleOCR / PaddleX logs |

## Output formats

### JSON

The JSON has these top-level keys:
- `source`: file name, sha256, page count, pages processed and pages that failed.
- `engine`: the Layout model, the Reader and the Judge (`layout_model`, `reader`, `judge`), the library versions
  and the device. Whenever the worker runs (MiniCPM, or the Judge) it also has the `torch`, `transformers`,
  `playwright` and `pillow` versions.
- `settings`: every option that affects the output, including `reader`, `judge` and `judge_threshold` (null
  with `--no-judge`).
- `postprocessing`: what the clean-up steps changed (with MiniCPM, also `repetition_cuts`; with the Judge, a
  `judge` summary: blocks judged, blocks below the threshold, a count per verdict, and how many blocks were
  judged before the document was known to be Traditional).
- `csv_tables`: which tables were exported and which were skipped.
- `elapsed_seconds`.
- `pages`.

Each page lists its `blocks`, each with these fields:
- `id`, `order`, `label`, `content`
- `bbox` and `polygon`, in pixels of the rendered page. `scale_px_per_pt` converts them back to PDF points.
- `in_markdown`
- with the Judge, on every text, formula and table block: `score` (the Judge's 0-1 rating of the written text, or
  `null`), `verdict` and `below_threshold`; charts, seals and figures have none of these (see "The Judge")

A page also keeps its unfiltered `layout_detections` with scores. When a clean-up step changes a block,
the original text is kept next to it (`content_raw` for the script repair, `content_before_reread` for
the re-read). That way no edit is hidden.

The JSON is always faithful to each page. A table that runs over a page break stays split into its
per-page parts.

### Markdown

The Markdown is built from the same blocks, with page furniture (headers, footers, page numbers) left
out. A table that continues on the next page is merged into one table, and heading levels are
re-assigned across the whole document.

### CSV

Each `table` block becomes `csv/pNNN_tableKK.csv`, where `NNN` is the page the table starts on and `KK`
is its position among that page's tables. The CSVs follow the Markdown, so a table merged across pages
is one CSV, and `--no-merge-tables` gives one CSV per page part.

- **Only well-formed tables are exported.** The table's HTML must parse into a rectangular grid. A
  table is skipped if it has text outside cells, rows of different widths, holes, spans that run past
  the edge, or nested tables, or if the model wrote something that is not an HTML table. Each skipped
  table is reported on stderr and listed in `csv_tables.skipped` in the JSON, with the reason.
- **Merged cells** (`rowspan` / `colspan`) are expanded by repeating the cell text in every cell they
  cover. Every row then has every value, which suits filtering and pivoting.
- **Side-by-side panels are stacked.** A long two-column table is often printed as several panels on a
  page, so its header repeats (年度 | 係數 | 年度 | 係數 | 年度 | 係數). Left as is, that header has
  duplicate column names, and anything that keys rows by name (`csv.DictReader`, most converters and
  importers) keeps only the last panel. When the first row is an exact repeat of a header whose cells
  are all different, the panels are stacked one after another into a single table, and dash or empty
  rows that pad the end of a panel are dropped. The Markdown and the JSON keep the printed layout.
- **Cell text:** `<br>` becomes a line break inside the cell, HTML entities are decoded and other
  whitespace is collapsed.
- **Encoding:** UTF-8 with a BOM, so Excel opens CJK text correctly.
- **Re-runs:** a run into the same folder first deletes the old `pNNN_tableKK.csv` files there. Other
  files in `csv/` are left alone.
- **JSON entries:** `csv_tables.exported` lists each file with its first page, block id, all the pages
  it covers, and its row and column counts (plus `panels` when side-by-side panels were stacked).

On the sample PDF, all 11 table blocks were valid and became 6 CSVs, because two tables run across
several pages. One is the drug list on pages 14–16 (93 rows). The other is the ICD-10 code table on
pages 17–20 (78 rows). In each, PaddleOCR's merge drops the header row repeated at the top of each
continuation page.

## What was done for accuracy

**The numbers and the steps in this section were measured with PaddleOCR-VL-1.6 (`--reader paddle`).**
The MiniCPM Reader is described after them.

The goal was the highest precision possible with these two models. Every default below was chosen by
scoring the output against the text layer of a 21-page Traditional Chinese insurance policy. The PDF's
own characters were assigned to layout blocks by position, so reading order does not affect the score.

| | Character error rate | Simplified chars in a Traditional doc |
|---|---|---|
| PaddleOCR-VL library defaults | 1.74 % | 188 |
| `pdf_ocr.py` | **0.40 %** | 18 |

Each change and the reason for it:

1. **Layout threshold 0.3 → 0.2.** At the default, the layout model silently dropped whole paragraphs.
   Lowering the threshold cut the characters lost to missing blocks from about 85 to about 25. At 0.1
   it starts inventing full-width boxes.
2. **Render at 216 DPI.** 288 DPI scored the same and is slower.
3. **Simplified → Traditional repair (`--script auto`).** The model drifts into Simplified forms when
   reading Traditional text (银行, 医院), and a higher resolution does not fix it. The repair runs only
   when the document is clearly Traditional: at least 30 Traditional-only characters, and no more than
   15 % of the script-specific characters Simplified. Even then it changes only characters that cannot
   be encoded in Big5 and that OpenCC maps to another form. A blanket OpenCC conversion would also
   rewrite valid Traditional characters such as 台→臺 and 后→後. Formula blocks are never touched.
4. **Truncated-block re-read.** Now and then the model returns a few characters for a whole paragraph,
   and it does so every time for the same input. The tool estimates from the ink how many glyphs a text
   block holds. If the recognised text has fewer than 35 % of that, the block is re-read from renders
   at 1.25× and 0.8×. The longer reading is kept only if it is clearly longer and still plausible. On
   the sample, this recovered a 77-character paragraph exactly.
5. **Table line breaks.** The model writes an in-cell line break as a literal `\n`. It is rewritten to
   `<br>`.
6. **Charts and seals are opt-in.** In a synthetic test, chart recognition invented data for a figure it
   took to be a chart. By default, charts and seals are saved as images instead.
7. **Cross-page restructuring only for the Markdown and CSV.** PaddleOCR's table merge moves a
   continued table's content into the first page's block and empties the later ones. Building the JSON
   from that would make whole pages look blank, so the JSON is built before the merge.
8. **Image names are prefixed with the page number.** PaddleOCR names crops by their bbox only, so crops
   from different pages could overwrite each other.
9. **`footnote` blocks stay in the Markdown.** PaddleOCR's default drops them, but they are real content.

The output is deterministic: running the same PDF twice gives identical Markdown and JSON content.

### The MiniCPM-OCR-1.0 Reader

MiniCPM-OCR-1.0 is the default Reader. The Layout model, block merging, reading order, cross-page table
merging and the Markdown, JSON and CSV outputs are the same for both Readers, because the PaddleOCR
pipeline still does all of that. It hands each cropped block to the worker instead of to PaddleOCR-VL
(see "How the worker is wired in" below). The worker reads one block at a time:

- It loads the model as a native transformers image-text-to-text model (no remote code) in bf16.
- The prompt depends on the block label: `Text Recognition:` for text-like blocks and inline formulas,
  `Table Recognition:`, `Formula Recognition:` for display formulas, `Chart to Table:` and
  `Seal Recognition:`. The image comes before the text, and thinking is off.
- Decoding is greedy with `no_repeat_ngram_size=100`.
- The token limit depends on the block: 4096 for text, 512 for titles, headers and footers, 2048 for formulas
  and references, 1024 for footnotes, 256 for seals and page numbers, 4096 for charts. For tables it is
  `--max-new-tokens`. A block that hits its limit cannot run away.
- Charts and seals are still opt-in (`--charts`, `--seals`).

The raw Reading is then cleaned in this order:

1. **Repetition cut.** A Reading that falls into a loop is cut back (ported from the official MiniCPM-OCR
   SDK). The JSON counts the cuts in `postprocessing.repetition_cuts`.
2. **OTSL to HTML.** MiniCPM writes tables in OTSL. They are converted with the same converter PaddleX uses.
3. **Table line breaks.** MiniCPM writes an in-cell line break as a *real* newline (PaddleOCR-VL writes a
   literal `\n`). Both become `<br>`, so the CSV keeps the line break inside the cell. This happens inside
   the Reader, so `postprocessing.tables_with_line_breaks_fixed` (the count of the literal `\n` fix) stays 0
   with MiniCPM.
4. **Simplified to Traditional repair**, as for PaddleOCR-VL. It is document-level and runs after all
   pages, which is why it comes after the Reader and not inside it.

The truncated-block re-read (`--no-recover`) also works with MiniCPM, but only without the Judge (`--no-judge`).

What differs from PaddleOCR-VL in the text: MiniCPM keeps the printed spacing around Latin letters and
digits (`表 1-1`, `OCR 及`), writes `[...]` for the printed `〔...〕`, and uses `(...)` for full-width
brackets inside table cells. These cost error-rate points when the text layer is the reference, though they
are cosmetic. It also misreads the odd character the repair cannot fix: `爰` came out as the Simplified `爱`,
which the repair turned into `愛`.

**Measurement (a stand-in, not the sample).** The 21-page insurance policy that the numbers above come from
was not available when MiniCPM was added. For a first check, pages 3 to 12 of a different Traditional
Chinese proposal PDF (digital-born, 5,961 text-layer characters, 10 pages) were run with `ocr_eval.py`:

| Reader | errors | character error rate |
|---|---|---|
| PaddleOCR-VL-1.6 | 0 | 0.00 % |
| MiniCPM-OCR-1.0 | 10 | 0.17 % |

This is only a smoke test of the harness and the Reader on a stand-in document. It is not a baseline and it
does not say which Reader is better on the sample. The measurement on the sample is a later ticket.
Roughly the same time per page for both Readers (about 65 s for these 10 pages, including the
model load).

### The Judge

The Judge is MiniCPM-OCR Jev. It does not read; it looks at the Block's image and at an image of the Reading
rendered with Chromium and KaTeX, and answers how likely the Reading is to be correct. That is the **Score**, from 0
to 1. It is on by default and works with either Reader. This version only scores: every Block is read once, and
nothing is read again because of a low Score. (Re-reading low-Score Blocks is the next step in the plan.)

- **What is judged.** Text, titles, footnotes, references (category `text`), inline and display formulas
  (`formula`) and tables (`table`). Charts, seals and figures are never judged (the Judge was not trained on
  them): they carry no `score`, `verdict` or `below_threshold`.
- **Per Block in the JSON.** `score` is the Judge's number for the *written* text. `verdict` says whether it is
  real. `below_threshold` is true when the Block should be looked at.

  | `verdict` | meaning | `score` | `below_threshold` |
  |---|---|---|---|
  | `scored` | a real Score | 0-1 | `score < --judge-threshold` |
  | `precheck_reject` | KaTeX cannot parse the Reading's LaTeX: the Reading is unusable | null | true |
  | `render_error` | the renderer failed: the Reading is unusable | null | true |
  | `missing_original_image` | the renderer got no readable crop: the Reading is unusable | null | true |
  | `encode_error` | the images could not be encoded: scoring never happened | null | true |
  | `forward_error` | the model (or the worker) failed: scoring never happened | null | true |

  The official SDK reports 0.0 for the first three and -1.0 for the last two; those are *not* Scores, so they
  are `null` here, and a failed Block is flagged rather than given a fake low Score. `verdict_detail` carries the
  reason. A failure on one Block never stops the run: the Reading is kept and the failure recorded. A page whose
  Judge call fails entirely gets `forward_error` on all its judged Blocks.
- **On stderr**, each page's line says how many of its judged Blocks are below the threshold, e.g.
  `[  2/21] page 4: 13 blocks, 9.9s, 13 of 13 judged below the threshold`, with the number not scored in
  brackets when there are some. The Markdown contains no Scores or markers.
- **Scores are per page.** The JSON is built before the cross-page table merge, so a Score never covers only half
  of a merged table.
- **Same text as written, give or take one space.** Before a Reading is judged it gets the same clean-up it gets for the output:
  the repetition cut, `\(...\)` and `\[...\]` rewritten to ` $ ... $ ` / ` $$ ... $$ ` (the pipeline does that
  anyway, afterwards; with the delimiters rewritten a Reading scored 0.9895 against 0.9951 for the raw `\(...\)`
  form on the SDK's text example, and both render as math), OTSL converted to HTML for tables, line breaks inside
  table cells turned into `<br>`, and the Simplified to Traditional repair (below). Raw OTSL scored about 0.0005 for
  a correct table, which is why tables go to the Judge as HTML. The one difference: the pipeline's rewrite turns the
  model's `\[ x \]` into `$$  x  $$` (two spaces, since the model already spaces its formulas), and the real Judge
  rated that 0.73 where `$$ x $$`, which renders the same in the Markdown, got 0.97 (SDK formula example). So the Judge
  is shown a formula with single spaces inside `$`/`$$`; the written text keeps what the pipeline writes.
- **One Chromium launch per batch.** The pipeline hands the Reader a page's Blocks in one call (one call per
  pixel-budget group, which on the stand-in was always one per page: four pages, four score requests), and the
  Readings of that call are rendered in one batch (a launch costs about 0.8 s) and scored one at a time (batching
  changes the Scores slightly). On the stand-in PDF the Judge added about 2.6 s per
  page (the four pages took 24 s of page time without it and 34 s with it).

**The repair is decided while reading, so the Judge sees repaired text.** The Simplified to Traditional repair is
document-level: it needs the whole document to decide the document is Traditional (at least 30 Traditional-only
characters and at most 15 % Simplified). The Judge works per Block, so the decision is made on the Readings seen so
far, in the order the pipeline hands the Blocks over, and stays made: once the rule holds, every later Reading is repaired before it is judged.
The Reading that completes the evidence is repaired too. Same input in the same order always gives the same
decision. The document-level repair still runs afterwards on the original Readings, exactly as before, so the
written text is unchanged by the Judge. Two consequences, both recorded in the JSON:

- A Block read *before* the decision was judged unrepaired, so its Score may describe the pre-repair text. Those
  Blocks have `judged_before_script_decision: true`. On the stand-in this was the first four Blocks of the first
  page; with a non-Traditional document the decision never happens and nothing is repaired.
- If the document-level repair ends up changing a Block differently from the sticky one (for example the sticky
  decision came early and the full survey disagrees), the block has `judged_text`, the text that was actually
  scored, next to `content`, the text that was written.

**Scores are not bit-for-bit reproducible across machines**, and the authors say thresholds must be calibrated on
your own data. On one machine and one GPU the same PDF gives identical Scores (the end-to-end test runs it twice
and compares the JSON).

**The default threshold of 0.98 flags most Blocks.** 0.98 is the authors' uncalibrated default (`tau`). On a
four-page run of a stand-in document (a digital-born Traditional Chinese proposal, *not* the sample), 40 judged
Blocks scored a median of 0.941 (range 0.485 to 0.994); 2 reached 0.98, 28 reached 0.90. The Readings were
plausible by eye, so most correct Readings are flagged until the threshold is calibrated on the sample (a
later ticket). `--judge-threshold` changes the flag; the Score itself does not depend on it. The two lowest
were a table (0.48; not checked against the page) and a paragraph where the Reader had misread `爰` as `爱` (0.72), a
known misread.

**Why the renderer needs the system's Noto Sans CJK, and how it is made reproducible.** The renderer is the
authors' own, vendored unchanged in `worker/render/` ([ADR-0002](docs/adr/0002-judge-uses-the-official-renderer.md)
and its amendment). The first plan was to bundle DroidSansFallbackFull *instead of* the system's Noto Sans CJK.
That does not work: Droid has no Latin letters, digits or Greek, so English text renders as empty boxes. In the
authors' setup Latin and digits come from Noto Sans CJK and Droid only fills CJK gaps. So the bundled Droid font
works *behind* the system's Noto Sans CJK. The renderer runs under a private fontconfig that lists only those two,
with the rendering rules (slight hinting, antialiasing, LCD filter) written out; they are not left to the machine,
because a file with the font folders alone changed the Scores (table example 0.025 instead of 0.032, the authors
observed 0.041). With the rules, Scores on the SDK's three reference examples deviate from the authors' by 0.0017 on
average and 0.009 at most (their own expectation is about 0.005). The full comparison is in the ADR.

### How the worker is wired in

The main command builds the PaddleOCR pipeline with the Layout model only (PaddleOCR-VL's weights are not
loaded for `--reader minicpm`). Then it replaces the pipeline's per-block recognizer with a bridge that
sends each crop to the worker as a PNG file in a temporary folder and reads the answer back as one line of
JSON. The bridge hands a page's Blocks to `block_reader.py`, "read a Block": a plain module (no GPU, no paddle)
that takes the Blocks, a Reader and a Judge, cleans and repairs each Reading, has the page judged in one batch,
and returns the final text with its Score, verdict and flag. The worker has two jobs over the same protocol: `read`
(MiniCPM) and `score` (render and judge a batch of Readings). With `--reader paddle` the bridge calls
PaddleOCR-VL's own model as the Reader and the worker only judges, so the worker loads no MiniCPM. The worker is a long-lived process: the model is loaded once, and the main command stops it at the
end of the run, whatever happens. The code is in `reader_bridge.py` (the hook), `worker_client.py` (the
process and protocol) and `worker/` (the torch side).

The hook uses three private PaddleX names. A start-up check verifies them and stops with a clear message
if a PaddleX upgrade has moved them (`uv sync` restores the pinned versions), and a test runs the hook
against the real pipeline.

If the worker crashes or stops answering, the page it was reading is recorded in `source.pages_failed`
(exit code 1) with the worker's last messages, and the next page starts a fresh worker. Every wait has a
timeout, so a stuck worker cannot hang the run.

## Caveats

- The tuning was measured on one Traditional Chinese document, with PaddleOCR-VL. Other languages and
  layouts should work, but these numbers do not carry over to them. MiniCPM-OCR-1.0 became the default
  before it was measured on that document; see the stand-in measurement above.
- The Judge adds three setup items (Chromium, `node`, Noto Sans CJK) and about 2.6 s per page; `--no-judge` avoids
  them. With the default threshold of 0.98 most Blocks are flagged (see "The Judge"), and a Score is only as
  comparable as the machine it was made on.
- `--reader minicpm` and the Judge need the worker environment (`uv sync --project worker`) and a CUDA GPU. The worker's
  torch wheel targets CUDA 13.2; a driver that supports only CUDA 13.0 needs a different torch index in
  `worker/pyproject.toml`. This was not tested.
- About 94 of the 23,390 characters in the sample are still wrong, and these are model errors:
  - a repeated 腫→腗 misread in the ICD tables
  - a few small blocks the layout model never detects
  - 干 / 于 left where 幹 / 於 was meant. They are valid Traditional characters, so the repair
    deliberately leaves them.
- The orientation and unwarping models are not in `models/`. Rotated or warped scans are therefore not
  corrected.
- If a PDF already has a real text layer, as the sample does, extracting that text directly is exact.
  OCR is what matters for scanned PDFs.
- Speed: about 3.5 minutes for the 21-page sample on an RTX 4070 Ti (12 GB), plus a few seconds to load
  the models.
- Some harmless messages still appear on stderr, such as `which: no ccache` and a "Bucketed
  engine_config" warning.

## Testing

```bash
uv sync                      # includes the pytest dev group
uv run pytest                # fast tests; the tests that need a PDF skip without one
PDF_OCR_SAMPLE_PDF=/path/to/policy.pdf uv run pytest          # also run the CLI on pages 1-3 (GPU + models)
```

The fast tests need neither a GPU nor models: "read a Block" with a fake Reader and a fake Judge (repair happens
before judging, unjudged labels are skipped, failure verdicts, `--no-judge` and the ink re-read), the sticky
repair, the Judge's verdicts with a fake renderer and model, the renderer's setup checks and fontconfig, the
bridges, the JSON fields, the prompt and token limit of each block label, the
clean-up of a Reading, the worker's request loop, the client's start-up, timeout and crash handling (against
a stand-in worker, `tests/fake_worker.py`), the bridge and the hook, and the `--reader` options.

The tests marked `gpu` need `PDF_OCR_SAMPLE_PDF` (a PDF with a real text layer; except the reference-example
test), the models, the worker environment (the Judge runs in it for either Reader) and the Judge's setup:

- `tests/test_hook_regression.py` runs the hook on a real page with the real Layout model and a fake Reader.
  It needs no Reader weights and little GPU memory, and it fails when a PaddleX upgrade moves the private
  names the hook uses.
- `tests/test_worker_crash.py` runs the whole CLI with a stand-in worker that crashes: the page it was reading
  is recorded as failed, and the next page gets a new worker.
- `tests/test_e2e.py` runs the CLI twice with the Judge on, plus once with `--no-judge`, for each Reader, on a few
  pages. It checks the output structure (the same for both Readers), that every judged Block has a `score`, a
  `verdict` and a `below_threshold` and charts, seals and figures have none, that stderr counts the Blocks below
  the threshold per page, that the Markdown has no Scores, that `--no-judge` gives no Scores and records
  `judge: false`, that the character error rate against the text layer stays at or below the bound in
  `tests/baseline.json`, and that both runs give identical Markdown and JSON, Scores included.
- `tests/test_judge_reference.py` needs no sample PDF: it starts the real worker, scores the SDK's three
  reference examples and compares with the Scores the authors observed (within 0.015), and checks that unparsable
  LaTeX and a missing crop get `precheck_reject` and `missing_original_image` while the rest of the batch is scored.
  It needs the worker environment, Chromium, `node` and Noto Sans CJK.

`PDF_OCR_PAGES` (default: the `pages` in `tests/baseline.json`, now `1-3`), `PDF_OCR_MODELS_DIR`,
`PDF_OCR_CER_BOUND`, `PDF_OCR_READERS` (default `minicpm,paddle`) and `PDF_OCR_WORKER_ENV` (the worker's
virtual environment; default `worker/.venv`) override the pages, the models folder, the bound, the Readers
to run and the worker environment. `PDF_OCR_WORKER_ENV` also works for the CLI itself.

`ocr_eval.py` is the error-rate harness. The PDF's characters are assigned to Blocks by position, so
reading order does not matter. It also runs standalone on a finished run and prints the error rate per
page and per Block label:

```bash
uv run ocr_eval.py input.pdf [output/<stem>/<stem>.json] [--report scores.json]
```

Text-layer characters that lie in no Block count as errors; figures, charts and seals are not scored;
whitespace, HTML markup, full/half-width forms and table-of-contents dot leaders are ignored.

The bound in `tests/baseline.json` is currently a **placeholder** (5 %), not a measurement. The baseline
of PaddleOCR-VL-1.6 on the 21-page sample still has to be measured with `ocr_eval.py` and recorded there,
on exactly the pages in `baseline.json` (the 0.40 % above came from an earlier throwaway harness with
different normalisation and the whole document, so it is not this tool's baseline).
