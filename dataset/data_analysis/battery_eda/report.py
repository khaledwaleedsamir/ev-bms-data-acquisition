"""
report.py
==========

Assembles everything the other modules computed into one automatic report,
covering: important findings, detected anomalies, data-quality assessment,
potential SOC features, potential SOH features, and recommendations for
future testing. Rendered as both Markdown and a self-contained HTML page.

`ReportBuilder` is a tiny generic document model (heading / text / table /
image / key-value block, in order) with two renderers - this avoids needing
an external Markdown-to-HTML library (none is installed in this
environment) while still producing a real HTML report, not just markdown
dumped into a <pre> tag.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from . import config


@dataclass
class ReportBuilder:
    title: str
    _blocks: list = field(default_factory=list)

    def add_heading(self, text: str, level: int = 2) -> None:
        self._blocks.append(("heading", level, text))

    def add_text(self, text: str) -> None:
        self._blocks.append(("text", text))

    def add_list(self, items: list[str]) -> None:
        self._blocks.append(("list", items))

    def add_table(self, df: pd.DataFrame, max_rows: int = 30) -> None:
        if df is None or df.empty:
            return
        self._blocks.append(("table", df.head(max_rows)))

    def add_key_values(self, values: dict) -> None:
        if not values:
            return
        self._blocks.append(("kv", values))

    def add_image(self, path: str, caption: str = "") -> None:
        if not path:
            return
        self._blocks.append(("image", path, caption))

    # -- rendering -----------------------------------------------------
    def render_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        for block in self._blocks:
            kind = block[0]
            if kind == "heading":
                _, level, text = block
                lines.append(f"{'#' * level} {text}")
                lines.append("")
            elif kind == "text":
                lines.append(block[1])
                lines.append("")
            elif kind == "list":
                lines.extend(f"- {item}" for item in block[1])
                lines.append("")
            elif kind == "kv":
                for k, v in block[1].items():
                    lines.append(f"- **{k}**: {_fmt_value(v)}")
                lines.append("")
            elif kind == "table":
                lines.append(block[1].to_markdown(index=False))
                lines.append("")
            elif kind == "image":
                _, path, caption = block
                lines.append(f"![{caption}]({_relative(path)})")
                if caption:
                    lines.append(f"*{caption}*")
                lines.append("")
        return "\n".join(lines)

    def render_html(self) -> str:
        parts = [f"<h1>{self.title}</h1>"]
        for block in self._blocks:
            kind = block[0]
            if kind == "heading":
                _, level, text = block
                parts.append(f"<h{level}>{text}</h{level}>")
            elif kind == "text":
                parts.append(f"<p>{block[1]}</p>")
            elif kind == "list":
                items = "".join(f"<li>{item}</li>" for item in block[1])
                parts.append(f"<ul>{items}</ul>")
            elif kind == "kv":
                rows = "".join(f"<tr><td><b>{k}</b></td><td>{_fmt_value(v)}</td></tr>"
                                for k, v in block[1].items())
                parts.append(f"<table class='kv'>{rows}</table>")
            elif kind == "table":
                parts.append(block[1].to_html(index=False, classes="data-table", border=0))
            elif kind == "image":
                _, path, caption = block
                rel = _relative(path)
                if path.endswith(".html"):
                    # Interactive Plotly figures are saved as standalone HTML
                    # - embed them inline via <iframe> so they stay interactive.
                    parts.append(
                        f"<iframe src='{rel}' width='100%' height='500' "
                        f"style='border:1px solid #ddd;'></iframe>"
                    )
                else:
                    parts.append(f"<img src='{rel}' style='max-width:100%;' />")
                if caption:
                    parts.append(f"<p class='caption'>{caption}</p>")
        body = "\n".join(parts)
        return f"<!doctype html><html><head><meta charset='utf-8'>" \
               f"<title>{self.title}</title>{_HTML_STYLE}</head><body>{body}</body></html>"

    def save(self, outdir: str, basename: str = "eda_report") -> dict:
        os.makedirs(outdir, exist_ok=True)
        md_path = os.path.join(outdir, f"{basename}.md")
        html_path = os.path.join(outdir, f"{basename}.html")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.render_markdown())
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self.render_html())
        return {"markdown": md_path, "html": html_path}


def _relative(path: str) -> str:
    # The report (report.md / report.html) is always saved directly in
    # `outdir`, and every figure is always saved in `outdir/figures/` (see
    # run_battery_eda.py) - so the link from the report to any figure is
    # always "figures/<filename>", regardless of the figure's absolute path.
    # Forward slash is used explicitly (not os.path.join) since this path is
    # embedded in Markdown/HTML, not passed to the filesystem - Windows
    # backslashes would not render as a valid link in either format.
    return f"figures/{os.path.basename(path)}"


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


_HTML_STYLE = """
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 1000px;
         margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }
  h1, h2, h3 { color: #14324d; }
  table.data-table, table.kv { border-collapse: collapse; margin: 1rem 0; width: 100%; }
  table.data-table th, table.data-table td, table.kv td {
      border: 1px solid #ddd; padding: 6px 10px; font-size: 0.9rem; text-align: left; }
  table.data-table th { background: #f2f5f8; }
  .caption { color: #666; font-size: 0.85rem; margin-top: -0.5rem; }
</style>
"""


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def generate_report(context: dict) -> ReportBuilder:
    """
    Build the full automatic report from everything run_battery_eda.py
    computed. `context` keys used (all optional except h5_path):

      h5_path, run_names, quality_report, summary_text, cleaning_log,
      anomaly_summary, anomaly_examples, usage_stats, cycles, segments,
      soc_ranking, soh_stats, dominant_frequencies, key_figures (dict of
      caption -> saved figure path), recommendations (list[str])
    """
    rb = ReportBuilder(title=f"Battery EDA Report - {os.path.basename(context.get('h5_path', ''))}")

    rb.add_text(
        f"Source file: `{context.get('h5_path', 'n/a')}` &nbsp;|&nbsp; "
        f"Runs analyzed: {context.get('run_names', 'all')}"
    )

    # --- Important findings -------------------------------------------------
    rb.add_heading("Important Findings")
    findings = context.get("findings", [])
    if findings:
        rb.add_list(findings)
    else:
        rb.add_text("No findings were explicitly flagged; see the sections below for details.")

    # --- Data quality assessment --------------------------------------------
    rb.add_heading("Data Quality Assessment")
    if context.get("quality_report") is not None:
        rb.add_text(f"<pre>{context['quality_report'].as_text()}</pre>")
    if context.get("cleaning_log"):
        rb.add_heading("Cleaning steps applied", level=3)
        rb.add_list(context["cleaning_log"])

    # --- Detected anomalies --------------------------------------------------
    rb.add_heading("Detected Anomalies")
    anomaly_summary = context.get("anomaly_summary")
    if anomaly_summary is not None and not anomaly_summary.empty:
        rb.add_table(anomaly_summary)
    else:
        rb.add_text("No anomaly summary was computed.")

    # --- Usage statistics ------------------------------------------------------
    if context.get("usage_stats"):
        rb.add_heading("Battery Usage Statistics")
        rb.add_key_values(context["usage_stats"])

    # --- Cycle / SOH summary ----------------------------------------------------
    if context.get("cycles") is not None:
        rb.add_heading("Detected Charge/Discharge Cycles")
        if context["cycles"].empty:
            rb.add_text("No full discharge->charge cycle was detected in the loaded run(s).")
        else:
            rb.add_table(context["cycles"])

    if context.get("soh_stats"):
        rb.add_heading("SOH Degradation Indicators")
        rb.add_key_values(context["soh_stats"])

    # --- Potential SOC features --------------------------------------------------
    rb.add_heading("Potential SOC Features")
    soc_ranking = context.get("soc_ranking")
    if soc_ranking is not None and not soc_ranking.empty:
        rb.add_text(
            "Columns ranked by |Pearson correlation| with the BMS-reported SOC "
            f"({config.SIGNAL_MAP['soc']}):"
        )
        rb.add_table(soc_ranking.rename("correlation_with_soc").reset_index()
                      .rename(columns={"index": "column"}))
    else:
        rb.add_text("Not enough numeric signal variety to rank SOC-correlated features.")

    # --- Potential SOH features --------------------------------------------------
    rb.add_heading("Potential SOH Features")
    soh_ranking = context.get("soh_ranking")
    if soh_ranking is not None and not soh_ranking.empty:
        rb.add_text("Cycle-level columns ranked by |Pearson correlation| with cycle_number:")
        rb.add_table(soh_ranking.rename("correlation_with_cycle_number").reset_index()
                      .rename(columns={"index": "column"}))
    else:
        rb.add_text(
            "Fewer than 3 full cycles were detected in this file/run selection, so "
            "cycle-level SOH feature ranking could not be computed. Candidate SOH "
            "indicators from domain knowledge and this dataset's own fields: "
            "discharge Ah per cycle (capacity fade), mean pulse resistance "
            "(resistance growth), cell voltage imbalance trend, and coulombic/energy "
            "efficiency."
        )

    # --- Key figures --------------------------------------------------------------
    if context.get("key_figures"):
        rb.add_heading("Key Figures")
        for caption, path in context["key_figures"].items():
            rb.add_image(path, caption=caption)

    # --- Recommendations ------------------------------------------------------------
    rb.add_heading("Recommendations for Future Testing")
    recommendations = context.get("recommendations") or _default_recommendations()
    rb.add_list(recommendations)

    return rb


def _default_recommendations() -> list[str]:
    return [
        "Log a reference/ground-truth SOH (e.g. a periodic full discharge capacity test "
        "against a calibrated load) - none exists anywhere in the current dataset, which "
        "blocks supervised SOH modeling.",
        "Increase logging rate for at least a subset of runs if switching-noise or "
        "fast transient analysis is needed - 1 Hz logging limits FFT resolution to "
        "sub-0.5 Hz phenomena.",
        "Add explicit rest periods (5-30 min) between charge and discharge segments in "
        "future protocols to get clean OCV-relaxation curves for SOC-OCV calibration.",
        "Investigate and fix the BLE/serial dropout events flagged in the data-quality "
        "section - large timestamp gaps reduce the usable continuous sequence length for "
        "sequence models (LSTM/GRU).",
        "Consider tagging runs with a consistent, parseable metadata schema (load, speed, "
        "ambient temperature) - several early runs have no attrs at all.",
    ]
