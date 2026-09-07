from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import torch

from .attention_probe import AttentionTrace
from .tokenize import TokenizedSentence


def _colour(value: float, maximum: float) -> str:
    fraction = 0.0 if maximum <= 0 else min(1.0, value / maximum)
    # White -> deep blue; a shared scale makes heads comparable.
    red = round(255 * (1.0 - fraction))
    green = round(255 * (1.0 - 0.75 * fraction))
    blue = 255
    return f"rgb({red},{green},{blue})"


def _matrix_table(labels: tuple[str, ...], matrix: torch.Tensor, maximum: float) -> str:
    rows: list[str] = ["<table><thead><tr><th class='corner'>query \\ key</th>"]
    rows.append("".join(f"<th title='{html.escape(label)}'>{html.escape(label)}</th>" for label in labels))
    rows.append("</tr></thead><tbody>")
    for row_index, row_label in enumerate(labels):
        cells = [f"<th title='{html.escape(row_label)}'>{html.escape(row_label)}</th>"]
        for column_index, value in enumerate(matrix[row_index].tolist()):
            if row_index < column_index:
                cells.append("<td class='masked'>—</td>")
                continue
            numeric = float(value)
            cells.append(
                f"<td style='background:{_colour(numeric, maximum)}' "
                f"title='query={html.escape(row_label)}, key={html.escape(labels[column_index])}'>"
                f"{numeric:.4f}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _json_safe_tensor(tensor: torch.Tensor) -> list[Any]:
    values = tensor.detach().cpu().tolist()

    def sanitize(value: Any) -> Any:
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value

    return sanitize(values)


def write_attention_artifacts(
    output_dir: str | Path,
    tokenized: TokenizedSentence,
    trace: AttentionTrace,
) -> tuple[Path, Path]:
    """Write an annotated token-level HTML map and machine-readable JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    probabilities = trace.probabilities[0].detach().cpu()
    num_heads = probabilities.shape[0]
    maximum = float(probabilities.max().item())
    panels = [("Mean across heads", probabilities.mean(dim=0))] + [
        (f"Head {head_index}", probabilities[head_index]) for head_index in range(num_heads)
    ]
    panel_html: list[str] = []
    for panel_index, (title, matrix) in enumerate(panels):
        display = "block" if panel_index == 0 else "none"
        panel_html.append(
            f"<section id='panel-{panel_index}' style='display:{display}'>"
            f"<h2>{html.escape(title)}</h2>{_matrix_table(tokenized.labels, matrix, maximum)}</section>"
        )

    options = "".join(f"<option value='{i}'>{html.escape(title)}</option>" for i, (title, _) in enumerate(panels))
    page = f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>Token attention map</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #172033; }}
table {{ border-collapse: collapse; font-size: 0.78rem; }}
th, td {{ border: 1px solid #c9d1e1; min-width: 4.2rem; height: 3rem; text-align: center; padding: .25rem; }}
thead th, tbody th {{ background: #eef2f8; position: sticky; }}
.corner {{ min-width: 7rem; }} .masked {{ background: #444; color: #fff; }}
.legend {{ display:inline-block; width:14rem; height:1rem; background:linear-gradient(to right,#fff,#0000ff); vertical-align:middle; }}
code {{ background:#eef2f8; padding:.1rem .3rem; }}
</style></head><body>
<h1>Token-level causal attention</h1>
<p><strong>Input:</strong> {html.escape(tokenized.text)}</p>
<p>Rows are query tokens; columns are key tokens. Gray cells are blocked by the causal mask.
Values are post-softmax attention probabilities. Labels use <code>␠</code> for a leading space.</p>
<p><span class='legend'></span> low → high probability (shared scale; maximum={maximum:.4f})</p>
<label for='head'>View: </label><select id='head'>{options}</select>
{''.join(panel_html)}
<script>
document.getElementById('head').addEventListener('change', function(event) {{
  document.querySelectorAll('section').forEach((section, index) => {{
    section.style.display = index === Number(event.target.value) ? 'block' : 'none';
  }});
}});
</script></body></html>"""
    html_file = output_path / "token_attention.html"
    html_file.write_text(page, encoding="utf-8")

    json_file = output_path / "token_attention.json"
    payload = {
        "text": tokenized.text,
        "token_ids": list(tokenized.token_ids),
        "token_bytes_utf8_hex": [token.hex() for token in tokenized.token_bytes],
        "labels": list(tokenized.labels),
        "shape": {"batch": 1, "heads": num_heads, "sequence": len(tokenized.token_ids)},
        "attention_probabilities": _json_safe_tensor(trace.probabilities[0]),
        "raw_scores": _json_safe_tensor(trace.raw_scores[0]),
    }
    json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return html_file, json_file
