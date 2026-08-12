"""Collect finished audit runs into a single PDF of metrics and figures.

    python make_report.py                        # every recognised run dir
    python make_report.py --dirs demo_a demo_b   # only these
    python make_report.py --glob "sweep_*"       # only matching dirs
    python make_report.py --out pdf_reports/x.pdf

Walks each run directory, reads its summary JSON, timing log and saved attack
results, and writes one section per run grouped by audit type. Output lands in
``pdf_reports/`` and nothing under the run directories is modified.

Report only: every number is copied from the run's own artifacts, and no
interpretation is added. Read the summary logs for that.

Three run types are recognised, by which summary file is present::

    report/urmia_summary.json         level-2 U-RMIA   (run_urmia.py)
    report/urmia_online_summary.json  level-3 U-RMIA   (run_urmia_online.py)
    report/ola_summary.json           OLA audit        (run_ola.py)

ROC curves are re-plotted here from ``report/exp/attack_result_*.npz`` rather
than embedding the PNGs the runs write. Two reasons: ``run_ola.py`` saves no
PNGs at all (``ola_utils.report_ola_attack`` writes only the archive), and
overlaying the roles on shared axes fits a comparison on one figure where the
per-role PNGs would take a page each -- an OLA run alone writes 36 of them.

Dependencies are matplotlib and numpy, both already required by the pipeline.
"""

import argparse
import glob as globlib
import json
import os
import textwrap
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

PAGE = (8.27, 11.69)  # A4 portrait, inches
MONO = {"family": "monospace", "fontsize": 7.4}
# Monospace characters that fit the text column at MONO's size. Lines longer
# than this get a proportionally smaller font rather than running off the page.
FIT_CHARS = 104
ROLE_COLORS = {
    "original": "#1f77b4", "unlearned": "#d62728", "retrained": "#2ca02c",
    "vanilla": "#1f77b4", "ola": "#d62728", "onion": "#2ca02c",
}
SUMMARY_FILES = {
    "urmia_summary.json": "level2",
    "urmia_online_summary.json": "level3",
    "ola_summary.json": "ola",
}
TYPE_TITLES = {
    "level2": "Level-2 U-RMIA (naive offline attack)",
    "level3": "Level-3 U-RMIA (strong attack, train-then-unlearn references)",
    "ola": "OLA (Outer Layer Attenuation)",
}
TYPE_ORDER = ["level2", "level3", "ola"]
# ola_utils.AUDIT_SUBSETS order: the outer layer is the defended set and the
# reason the run exists, so it leads. Alphabetical would bury it under
# "all_members".
SUBSET_ORDER = ["outer", "inner", "all_members"]


def ordered_subsets(names):
    known = [s for s in SUBSET_ORDER if s in names]
    return known + sorted(n for n in names if n not in SUBSET_ORDER)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def classify(run_dir):
    """Return ``(type, summary_path)`` for a run directory, or ``(None, None)``."""
    for filename, kind in SUMMARY_FILES.items():
        path = os.path.join(run_dir, "report", filename)
        if os.path.exists(path):
            return kind, path
    return None, None


def read_timings(run_dir):
    """The timing lines a run logs, as ``(label, value)`` pairs.

    Read from the log rather than recomputed: these are the only record of how
    long a stage took, and a resumed run legitimately reports a short training
    time because it loaded most models from disk.
    """
    path = os.path.join(run_dir, "report", "log_time_analysis.log")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            low = line.lower()
            if " took " in low or "total runtime" in low or "ready in" in low:
                message = line.split("INFO", 1)[-1].strip(" \t-")
                rows.append(message.strip())
    # Later lines win: a resumed run appends a second pass to the same file.
    seen = {}
    for row in rows:
        key = row.split(" took ")[0].split(" ready in ")[0]
        seen[key] = row
    return list(seen.values())


def load_attack_results(run_dir):
    """``{name: {fpr, tpr, auc}}`` for every saved attack archive in the run."""
    pattern = os.path.join(run_dir, "report", "exp", "attack_result_*.npz")
    out = {}
    for path in sorted(globlib.glob(pattern)):
        name = os.path.basename(path)[len("attack_result_"):-len(".npz")]
        try:
            with np.load(path) as archive:
                out[name] = {
                    "fpr": archive["fpr"],
                    "tpr": archive["tpr"],
                    "auc": float(archive["auc"]),
                }
        except (OSError, ValueError, KeyError):
            continue  # a partial write from an interrupted run
    return out


def load_run(run_dir):
    kind, summary_path = classify(run_dir)
    if kind is None:
        return None
    with open(summary_path, "r") as f:
        summary = json.load(f)
    return {
        "dir": run_dir,
        "type": kind,
        "summary": summary,
        "timings": read_timings(run_dir),
        "curves": load_attack_results(run_dir),
    }


# --------------------------------------------------------------------------
# text / table helpers
# --------------------------------------------------------------------------
def fmt(value, width=9, places=4):
    """Fixed-width cell. ``None`` becomes 'n/a' rather than vanishing."""
    if value is None:
        return f"{'n/a':>{width}}"
    if isinstance(value, bool):
        return f"{str(value):>{width}}"
    if isinstance(value, (int, np.integer)):
        return f"{value:>{width}d}"
    if isinstance(value, float):
        return f"{value:>{width}.{places}f}"
    return f"{str(value):>{width}}"


def table_lines(headers, rows, first_width=14, width=11, places=4):
    """Render a table as monospace lines: header, rule, then one line per row.

    Columns are joined with an explicit space, and ``width`` is widened to the
    longest header, so a header at least as long as its cells (``forget_acc``
    against ``0.6540``) cannot run into its neighbour.
    """
    width = max(width, max((len(str(h)) for h in headers[1:]), default=0) + 1)
    head = f"{headers[0]:<{first_width}}" + " ".join(
        f"{h:>{width}}" for h in headers[1:]
    )
    lines = [head, "-" * len(head)]
    for row in rows:
        line = f"{str(row[0]):<{first_width}}" + " ".join(
            fmt(c, width, places) for c in row[1:]
        )
        lines.append(line)
    return lines


class Pager:
    """Sequential text/figure writer that starts a new page when one fills up."""

    def __init__(self, pdf):
        self.pdf = pdf
        self.ax = None
        self.y = 0.0

    def _new_page(self):
        self.flush()
        fig = plt.figure(figsize=PAGE)
        self.ax = fig.add_axes([0.06, 0.04, 0.90, 0.92])
        self.ax.axis("off")
        self.y = 1.0

    def flush(self):
        if self.ax is not None:
            self.pdf.savefig(self.ax.figure)
            plt.close(self.ax.figure)
            self.ax = None

    def space(self, amount):
        """Reserve vertical room, starting a page if this block will not fit."""
        if self.ax is None or self.y - amount < 0.0:
            self._new_page()

    def text(self, content, dy=0.016, **kwargs):
        style = dict(MONO)
        style.update(kwargs)
        self.ax.text(0.0, self.y, content, va="top", ha="left",
                     transform=self.ax.transAxes, **style)
        self.y -= dy

    def heading(self, title, subtitle=None, level=1):
        size = {1: 15, 2: 11.5}[level]
        self.space(0.10 if level == 1 else 0.07)
        if level == 1:
            self.y -= 0.012
        self.text(title, dy=0.028, family="sans-serif", fontsize=size,
                  fontweight="bold", color="#111111")
        if subtitle:
            self.text(subtitle, dy=0.020, family="sans-serif", fontsize=8.5,
                      color="#555555")
        self.y -= 0.006

    def block(self, lines, dy=0.0135, **kwargs):
        """Write lines together, keeping them on one page where possible.

        The font shrinks to fit the widest line. A run-comparison table grows a
        column per scorer, and at a fixed size the level-3 six-scorer table runs
        off the right edge -- silently, because nothing clips it.
        """
        longest = max((len(line) for line in lines), default=0)
        if longest > FIT_CHARS:
            kwargs.setdefault("fontsize",
                              max(4.6, MONO["fontsize"] * FIT_CHARS / longest))
        self.space(min(0.95, dy * (len(lines) + 1)))
        for line in lines:
            if self.y < 0.0:
                self._new_page()
            self.text(line, dy=dy, **kwargs)

    def caption(self, text_content):
        for line in textwrap.wrap(text_content, 108):
            self.text(line, dy=0.0145, family="sans-serif", fontsize=7.6,
                      color="#555555")


# --------------------------------------------------------------------------
# per-type sections
# --------------------------------------------------------------------------
def unlearn_block_lines(summary):
    """The ``unlearn`` config the run recorded in its own summary."""
    block = summary.get("unlearn", {})
    if not block:
        return []
    params = block.get("params") or {}
    lines = [
        f"algorithm    : {block.get('algorithm')}",
        f"forget_size  : {block.get('forget_size')}",
    ]
    if params:
        rendered = ", ".join(f"{k}={v}" for k, v in params.items())
        for i, chunk in enumerate(textwrap.wrap(rendered, 92)):
            lines.append(f"{'params' if i == 0 else '':<13}: {chunk}"
                         if i == 0 else f"{'':<15}{chunk}")
    return lines


def positive_control_lines(control):
    """Flatten whichever positive-control shape the run wrote."""
    if not control:
        return []
    lines = []
    for key in ("dead_scorers", "weak_ceiling_scorers"):
        if key in control:
            value = control[key] or "none"
            lines.append(f"{key:<22}: {value}")
    for key in ("null_se", "chance_cut", "dead_threshold", "elevated_threshold",
                "min_ceiling_auc"):
        if key in control and control[key] is not None:
            lines.append(f"{key:<22}: {control[key]:.5f}")
    aucs = control.get("vanilla_auc") or control.get("original_two_sided_auc")
    if aucs:
        label = "vanilla" if "vanilla_auc" in control else "original (two-sided)"
        lines.append(f"{label + ' AUC':<22}: "
                     + ", ".join(f"{k}={v:.4f}" for k, v in aucs.items()))
    return lines


METRIC_COLUMNS = [
    ("auc", "AUC", 4),
    ("two_sided_auc", "2-sided", 4),
    ("one_fpr", "TPR@1%", 4),
    ("one_tenth_fpr", "TPR@.1%", 4),
    ("zero_fpr", "TPR@0%", 4),
]


def section_level2(pager, run):
    roles = run["summary"].get("roles", {})
    headers = ["role"] + [label for _, label, _ in METRIC_COLUMNS] + [
        "test_acc", "forget_acc"]
    rows = []
    for role, values in roles.items():
        rows.append([role]
                    + [values.get(key) for key, _, _ in METRIC_COLUMNS]
                    + [values.get("test_acc"), values.get("forget_acc")])
    pager.block(table_lines(headers, rows, first_width=12, width=10))


def scorer_tables(pager, roles, metric, label):
    """One table: rows are roles, columns are scorers, cells are ``metric``."""
    scorers = sorted({s for v in roles.values() for s in v.get("attacks", {})})
    if not scorers:
        return
    width = max(11, max(len(s) for s in scorers) + 2)
    rows = [[role] + [values.get("attacks", {}).get(s, {}).get(metric)
                      for s in scorers]
            for role, values in roles.items()]
    pager.text(label, dy=0.017, family="sans-serif", fontsize=8.5,
               fontweight="bold")
    pager.block(table_lines(["role"] + scorers, rows, first_width=12,
                            width=width))
    pager.y -= 0.008


def section_level3(pager, run):
    roles = run["summary"].get("roles", {})
    acc_rows = [[role, values.get("test_acc"), values.get("forget_acc")]
                for role, values in roles.items()]
    pager.text("Accuracy", dy=0.017, family="sans-serif", fontsize=8.5,
               fontweight="bold")
    pager.block(table_lines(["role", "test_acc", "forget_acc"], acc_rows,
                            first_width=12, width=12))
    pager.y -= 0.008
    for metric, label in [("auc", "AUC"), ("two_sided_auc", "Two-sided AUC"),
                          ("one_tenth_fpr", "TPR @ 0.1% FPR"),
                          ("one_fpr", "TPR @ 1% FPR")]:
        scorer_tables(pager, roles, metric, label)


def section_ola(pager, run):
    summary = run["summary"]
    roles = summary.get("roles", {})

    acc_keys = sorted({k for v in roles.values()
                       for k in (v.get("accuracy") or {})
                       if isinstance((v.get("accuracy") or {}).get(k),
                                     (int, float))})
    if acc_keys:
        rows = [[role] + [(values.get("accuracy") or {}).get(k) for k in acc_keys]
                for role, values in roles.items()]
        pager.text("Accuracy", dy=0.017, family="sans-serif", fontsize=8.5,
                   fontweight="bold")
        pager.block(table_lines(["role"] + acc_keys, rows, first_width=12,
                                width=max(11, max(len(k) for k in acc_keys) + 2)))
        pager.y -= 0.008

    subsets = ordered_subsets({s for v in roles.values()
                               for s in v.get("attacks", {})})
    for subset in subsets:
        scorers = sorted({sc for v in roles.values()
                          for sc in v.get("attacks", {}).get(subset, {})})
        if not scorers:
            continue
        width = max(11, max(len(s) for s in scorers) + 2)
        pager.text(f"Subset: {subset}", dy=0.019, family="sans-serif",
                   fontsize=9.5, fontweight="bold", color="#333333")
        for metric, label in [("auc", "AUC"), ("two_sided_auc", "Two-sided AUC"),
                              ("one_tenth_fpr", "TPR @ 0.1% FPR")]:
            rows = [[role]
                    + [values.get("attacks", {}).get(subset, {})
                       .get(sc, {}).get(metric) for sc in scorers]
                    for role, values in roles.items()]
            pager.text(f"  {label}", dy=0.016, family="sans-serif", fontsize=8.2)
            pager.block(table_lines(["role"] + scorers, rows, first_width=12,
                                    width=width))
            pager.y -= 0.006
        pager.y -= 0.004

    excess = summary.get("excess_over_onion")
    if excess:
        pager.text("Excess over the onion floor (two-sided AUC difference)",
                   dy=0.018, family="sans-serif", fontsize=9.5,
                   fontweight="bold", color="#333333")
        for subset in ordered_subsets(excess):
            scorers = sorted(excess[subset])
            width = max(11, max(len(s) for s in scorers) + 2)
            rows = [
                ["excess"] + [excess[subset][s]["excess"] for s in scorers],
                ["in null SE"] + [excess[subset][s]["excess_null_se"]
                                  for s in scorers],
            ]
            null_se = excess[subset][scorers[0]]["null_se"]
            pager.text(f"  {subset}  (null SE {null_se:.5f})", dy=0.016,
                       family="sans-serif", fontsize=8.2)
            pager.block(table_lines(["", *scorers], rows, first_width=12,
                                    width=width))
            pager.y -= 0.006


SECTION_RENDERERS = {"level2": section_level2, "level3": section_level3,
                     "ola": section_ola}


# --------------------------------------------------------------------------
# cross-run comparison
# --------------------------------------------------------------------------
def _run_label(run):
    """Identifier for a run in a cross-run table.

    The directory name, not dataset+algorithm: two runs of one algorithm on one
    dataset differ only by config (a params sweep), and collapsing them to the
    same label makes the comparison table unreadable. Prefixes that are constant
    within the table -- the ``sweep_`` convention and the level, which is the
    section heading -- are dropped to buy width for the metric columns.
    """
    name = os.path.basename(run["dir"])
    for prefix in ("sweep_", "demo_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for prefix in ("l2_", "l3_", "ola_", "level2_", "level3_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name[:24]


def overview_level2(group):
    headers = ["run", "orig AUC", "unl AUC", "unl 2-sid", "retr AUC",
               "unl test", "unl forget", "retr forget"]
    rows = []
    for run in group:
        roles = run["summary"].get("roles", {})
        original = roles.get("original", {})
        unlearned = roles.get("unlearned", {})
        retrained = roles.get("retrained", {})
        rows.append([_run_label(run), original.get("auc"), unlearned.get("auc"),
                     unlearned.get("two_sided_auc"), retrained.get("auc"),
                     unlearned.get("test_acc"), unlearned.get("forget_acc"),
                     retrained.get("forget_acc")])
    return headers, rows


def overview_level3(group):
    """Two-sided AUC on the 'unlearned' role, per scorer, one row per run."""
    scorers = sorted({s for run in group
                      for s in run["summary"].get("roles", {})
                      .get("unlearned", {}).get("attacks", {})})
    headers = ["run"] + scorers + ["unl test", "unl forget"]
    rows = []
    for run in group:
        unlearned = run["summary"].get("roles", {}).get("unlearned", {})
        attacks = unlearned.get("attacks", {})
        rows.append([_run_label(run)]
                    + [attacks.get(s, {}).get("two_sided_auc") for s in scorers]
                    + [unlearned.get("test_acc"), unlearned.get("forget_acc")])
    return headers, rows


def overview_ola(group):
    """Outer-subset two-sided AUC per role, per scorer, one row per run/role."""
    scorers = sorted({sc for run in group
                      for role in run["summary"].get("roles", {}).values()
                      for sc in role.get("attacks", {}).get("outer", {})})
    headers = ["run / role"] + scorers
    rows = []
    for run in group:
        for role, values in run["summary"].get("roles", {}).items():
            outer = values.get("attacks", {}).get("outer", {})
            rows.append([f"{_run_label(run)}/{role}"[:26]]
                        + [outer.get(s, {}).get("two_sided_auc") for s in scorers])
    return headers, rows


OVERVIEWS = {"level2": overview_level2, "level3": overview_level3,
             "ola": overview_ola}
OVERVIEW_CAPTIONS = {
    "level2": "One row per run. 'unl' is the unlearned model, 'retr' the "
              "retrained one. Two-sided AUC is max(AUC, 1-AUC).",
    "level3": "Two-sided AUC on the unlearned model, one column per scorer.",
    "ola": "Two-sided AUC on the OUTER subset (the attenuated points), one row "
           "per role.",
}


def overview_section(pager, kind, group):
    headers, rows = OVERVIEWS[kind](group)
    if len(headers) < 2 or not rows:
        return
    pager.text("Comparison across runs", dy=0.019, family="sans-serif",
               fontsize=10, fontweight="bold")
    pager.caption(OVERVIEW_CAPTIONS[kind])
    pager.y -= 0.006
    pager.block(table_lines(headers, rows, first_width=25, width=9))
    pager.y -= 0.012


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def curve_groups(run):
    """Group saved curves into one plot each: ``{title: {role: curve}}``.

    Names are ``<role>`` (level 2), ``<role>_<scorer>`` (level 3) or
    ``<role>_<subset>_<scorer>`` (OLA), so the role prefix is what varies within
    a plot and the remainder is the plot's identity.
    """
    roles = list(run["summary"].get("roles", {}))
    groups = {}
    for name, curve in run["curves"].items():
        role = next((r for r in roles
                     if name == r or name.startswith(r + "_")), None)
        if role is None:
            continue
        rest = name[len(role):].lstrip("_") or "attack"
        groups.setdefault(rest, {})[role] = curve
    return dict(sorted(groups.items()))


def draw_roc_pages(pdf, run, per_page=6):
    groups = curve_groups(run)
    if not groups:
        return 0
    titles = list(groups)
    pages = 0
    for start in range(0, len(titles), per_page):
        chunk = titles[start:start + per_page]
        cols = 1 if len(chunk) == 1 else 2
        rows = -(-len(chunk) // cols)
        # Height follows the row count instead of always filling A4: a level-2
        # run has a single curve group, and stretching one plot over a full page
        # distorts a ROC into an unreadable ribbon.
        height = min(PAGE[1], 1.0 + 3.6 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=(PAGE[0], height),
                                 squeeze=False)
        fig.suptitle(f"{os.path.basename(run['dir'])} — log-scale ROC",
                     fontsize=11, fontweight="bold")
        for ax, title in zip(axes.ravel(), chunk):
            for role, curve in sorted(groups[title].items()):
                ax.plot(curve["fpr"], curve["tpr"], linewidth=1.2,
                        color=ROLE_COLORS.get(role),
                        label=f"{role} ({curve['auc']:.3f})")
            ax.plot([1e-5, 1], [1e-5, 1], "--", color="#999999", linewidth=0.8,
                    label="chance")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-5, 1)
            ax.set_ylim(1e-5, 1)
            ax.set_xlabel("FPR", fontsize=7)
            ax.set_ylabel("TPR", fontsize=7)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3, linewidth=0.4)
            ax.legend(fontsize=5.6, loc="upper left")
            ax.set_box_aspect(1)  # a ROC is only readable square
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.tight_layout(rect=[0, 0, 1, 1.0 - 0.5 / height])
        pdf.savefig(fig)
        plt.close(fig)
        pages += 1
    return pages


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def cover_page(pdf, runs, skipped):
    fig = plt.figure(figsize=PAGE)
    ax = fig.add_axes([0.06, 0.04, 0.90, 0.92])
    ax.axis("off")
    y = 1.0
    ax.text(0.0, y, "Membership-inference audit report", fontsize=19,
            fontweight="bold", va="top", transform=ax.transAxes)
    y -= 0.045
    ax.text(0.0, y, datetime.now().strftime("Generated %Y-%m-%d %H:%M"),
            fontsize=9, color="#555555", va="top", transform=ax.transAxes)
    y -= 0.045
    ax.text(0.0, y, f"{len(runs)} run(s) collected", fontsize=9,
            color="#555555", va="top", transform=ax.transAxes)
    y -= 0.04

    headers = ["run directory", "type", "algorithm", "dataset*", "roles", "curves"]
    rows = []
    for run in runs:
        summary = run["summary"]
        algorithm = (summary.get("unlearn") or {}).get("algorithm", "—")
        if run["type"] == "ola":
            ola = summary.get("ola") or {}
            algorithm = f"ola p_gen={ola.get('p_gen')} p_mem={ola.get('p_mem')}"
        rows.append([os.path.basename(run["dir"]), run["type"], algorithm,
                     guess_dataset(run["dir"]),
                     len(summary.get("roles", {})), len(run["curves"])])
    line = (f"{headers[0]:<34}{headers[1]:<9}{headers[2]:<34}"
            f"{headers[3]:<10}{headers[4]:>6}{headers[5]:>8}")
    ax.text(0.0, y, line, va="top", transform=ax.transAxes, **MONO)
    y -= 0.016
    ax.text(0.0, y, "-" * len(line), va="top", transform=ax.transAxes, **MONO)
    y -= 0.016
    for row in rows:
        ax.text(0.0, y,
                f"{str(row[0])[:33]:<34}{row[1]:<9}{str(row[2])[:33]:<34}"
                f"{str(row[3]):<10}{row[4]:>6}{row[5]:>8}",
                va="top", transform=ax.transAxes, **MONO)
        y -= 0.016

    y -= 0.02
    ax.text(0.0, y, "* dataset is read from the run directory name, not the "
                    "summary (the summaries do not record it).",
            fontsize=7.4, color="#777777", va="top", transform=ax.transAxes)
    y -= 0.03
    if skipped:
        ax.text(0.0, y, f"Skipped {len(skipped)} directory/ies with no summary "
                        f"file (unfinished or not an audit run):",
                fontsize=8, color="#a33", va="top", transform=ax.transAxes)
        y -= 0.02
        for path in skipped[:18]:
            ax.text(0.02, y, os.path.basename(path), va="top",
                    transform=ax.transAxes, **MONO)
            y -= 0.015
    pdf.savefig(fig)
    plt.close(fig)


def guess_dataset(run_dir):
    name = os.path.basename(run_dir).lower()
    for candidate in ("cifar100", "cifar10", "purchase100", "texas100", "agnews"):
        if candidate in name:
            return candidate
    return "—"


def build(runs, skipped, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with PdfPages(out_path) as pdf:
        cover_page(pdf, runs, skipped)
        for kind in TYPE_ORDER:
            group = [r for r in runs if r["type"] == kind]
            if not group:
                continue
            pager = Pager(pdf)
            pager.space(0.2)
            pager.heading(TYPE_TITLES[kind], f"{len(group)} run(s)", level=1)
            overview_section(pager, kind, group)
            pager.flush()
            for run in group:
                pager = Pager(pdf)
                pager.space(0.3)
                pager.heading(os.path.basename(run["dir"]),
                              f"{TYPE_TITLES[kind]} · dataset "
                              f"{guess_dataset(run['dir'])}", level=2)

                config_lines = unlearn_block_lines(run["summary"])
                if run["type"] == "ola" and run["summary"].get("ola"):
                    config_lines = [f"{k:<13}: {v}"
                                    for k, v in run["summary"]["ola"].items()]
                if config_lines:
                    pager.text("Configuration", dy=0.017, family="sans-serif",
                               fontsize=8.5, fontweight="bold")
                    pager.block(config_lines)
                    pager.y -= 0.008

                SECTION_RENDERERS[kind](pager, run)

                control = run["summary"].get("positive_control")
                control_lines = positive_control_lines(control)
                if control_lines:
                    pager.y -= 0.004
                    pager.text("Positive control", dy=0.017,
                               family="sans-serif", fontsize=8.5,
                               fontweight="bold")
                    pager.block(control_lines)

                if run["timings"]:
                    pager.y -= 0.004
                    pager.text("Timing", dy=0.017, family="sans-serif",
                               fontsize=8.5, fontweight="bold")
                    pager.block(run["timings"])
                pager.flush()
                draw_roc_pages(pdf, run)

        info = pdf.infodict()
        info["Title"] = "Membership-inference audit report"
        info["Subject"] = f"{len(runs)} audit runs"
        info["CreationDate"] = datetime.now()


def main():
    parser = argparse.ArgumentParser(
        description="Collect finished audit runs into one PDF.")
    parser.add_argument("--dirs", nargs="*", default=None,
                        help="Run directories. Default: scan the working dir.")
    parser.add_argument("--glob", default="*",
                        help="Pattern for the scan (default: every directory).")
    parser.add_argument("--out", default=None,
                        help="Output PDF (default: pdf_reports/audit_report_"
                             "<timestamp>.pdf).")
    args = parser.parse_args()

    if args.dirs:
        candidates = args.dirs
    else:
        candidates = sorted(d for d in globlib.glob(args.glob) if os.path.isdir(d))

    runs, skipped = [], []
    for candidate in candidates:
        run = load_run(candidate)
        if run is None:
            if os.path.isdir(os.path.join(candidate, "report")):
                skipped.append(candidate)
            continue
        runs.append(run)

    if not runs:
        print("No finished audit runs found. Looked for "
              + ", ".join(f"<dir>/report/{f}" for f in SUMMARY_FILES)
              + f"\nScanned {len(candidates)} director(y/ies).")
        return 1

    runs.sort(key=lambda r: (TYPE_ORDER.index(r["type"]), r["dir"]))
    out_path = args.out or os.path.join(
        "pdf_reports",
        f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    build(runs, skipped, out_path)

    print(f"Wrote {out_path}")
    for kind in TYPE_ORDER:
        group = [r for r in runs if r["type"] == kind]
        if group:
            print(f"  {TYPE_TITLES[kind]}: {len(group)}")
            for run in group:
                print(f"    - {run['dir']} "
                      f"({len(run['curves'])} attack archives)")
    if skipped:
        print(f"  skipped (no summary): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
