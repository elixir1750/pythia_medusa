from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from pythia_medusa.utils.io import write_json


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric:
        return None
    return numeric


def _extract_points(rows: list[dict[str, Any]], *, x_key: str, y_key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x_value = _safe_float(row.get(x_key))
        y_value = _safe_float(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        points.append((x_value, y_value))
    return points


def _line_points(
    points: list[tuple[float, float]],
    *,
    width: int,
    height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    padding: int = 40,
) -> str:
    if not points:
        return ""

    min_x, max_x = x_range
    min_y, max_y = y_range
    inner_width = max(width - padding * 2, 1)
    inner_height = max(height - padding * 2, 1)

    if max_x == min_x:
        max_x = min_x + 1.0
    if max_y == min_y:
        max_y = min_y + 1.0

    svg_points: list[str] = []
    for x_value, y_value in points:
        x_pos = padding + ((x_value - min_x) / (max_x - min_x)) * inner_width
        y_pos = height - padding - ((y_value - min_y) / (max_y - min_y)) * inner_height
        svg_points.append(f"{x_pos:.2f},{y_pos:.2f}")
    return " ".join(svg_points)


def _build_chart(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    width: int = 860,
    height: int = 280,
) -> str:
    non_empty = [entry for entry in series if entry["points"]]
    if not non_empty:
        return (
            '<section class="chart-card">'
            f"<h3>{html.escape(title)}</h3>"
            '<p class="empty">No data available yet.</p>'
            "</section>"
        )

    xs = [point[0] for entry in non_empty for point in entry["points"]]
    ys = [point[1] for entry in non_empty for point in entry["points"]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_y == max_y:
        min_y -= 0.5
        max_y += 0.5
    else:
        margin = (max_y - min_y) * 0.08
        min_y -= margin
        max_y += margin

    grid_lines: list[str] = []
    padding = 40
    inner_height = max(height - padding * 2, 1)
    for index in range(5):
        y = padding + (inner_height / 4.0) * index
        grid_lines.append(
            f'<line x1="{padding}" y1="{y:.2f}" x2="{width - padding}" y2="{y:.2f}" class="grid" />'
        )

    polylines: list[str] = []
    legend_items: list[str] = []
    for entry in non_empty:
        points = _line_points(
            entry["points"],
            width=width,
            height=height,
            x_range=(min_x, max_x),
            y_range=(min_y, max_y),
            padding=padding,
        )
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="{entry["color"]}" '
            'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />'
        )
        legend_items.append(
            '<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{entry["color"]};"></span>'
            f"{html.escape(entry['label'])}"
            "</span>"
        )

    return (
        '<section class="chart-card">'
        f"<h3>{html.escape(title)}</h3>"
        f'<div class="legend">{"".join(legend_items)}</div>'
        f'<svg viewBox="0 0 {width} {height}" class="chart-svg" role="img" '
        f'aria-label="{html.escape(title)}">'
        f'{"".join(grid_lines)}'
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" '
        f'y2="{height - padding}" class="axis" />'
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" class="axis" />'
        f'{"".join(polylines)}'
        f'<text x="{width / 2:.2f}" y="{height - 8}" class="axis-label">{html.escape(x_label)}</text>'
        f'<text x="16" y="{height / 2:.2f}" class="axis-label" transform="rotate(-90 16 {height / 2:.2f})">'
        f"{html.escape(y_label)}</text>"
        f'<text x="{padding}" y="18" class="axis-tick">{min_y:.4f}</text>'
        f'<text x="{width - padding - 12}" y="18" class="axis-tick">{max_y:.4f}</text>'
        f'<text x="{padding}" y="{height - padding + 20}" class="axis-tick">{min_x:.0f}</text>'
        f'<text x="{width - padding - 12}" y="{height - padding + 20}" class="axis-tick">{max_x:.0f}</text>'
        "</svg>"
        "</section>"
    )


def _summary_cards(run_payload: dict[str, Any], summary: dict[str, Any]) -> str:
    parameter_stats = run_payload.get("parameter_stats", {})
    final_train = summary.get("final_train_metrics", {})
    final_valid = summary.get("final_valid_metrics", {})
    cards = [
        ("Trainable Params", f"{int(parameter_stats.get('trainable_params', 0)):,}"),
        ("Trainable %", f"{parameter_stats.get('trainable_pct', 0.0) * 100:.2f}%"),
        ("Final Train Loss", f"{_safe_float(final_train.get('loss')) or 0.0:.4f}"),
    ]
    if final_valid:
        cards.append(("Final Valid Loss", f"{_safe_float(final_valid.get('loss')) or 0.0:.4f}"))
    return "".join(
        '<div class="summary-card">'
        f'<div class="summary-label">{html.escape(label)}</div>'
        f'<div class="summary-value">{html.escape(value)}</div>'
        "</div>"
        for label, value in cards
    )


def _config_table(config: dict[str, Any]) -> str:
    rows = []
    for key, value in config.items():
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))}</td>"
            "</tr>"
        )
    return '<table class="config-table">' + "".join(rows) + "</table>"


def build_training_dashboard_html(run_payload: dict[str, Any]) -> str:
    summary = run_payload.get("summary", {})
    history = summary.get("history", [])
    eval_history = summary.get("eval_history", [])
    config = run_payload.get("config", {})

    chart_specs = [
        _build_chart(
            title="Training Total Loss",
            x_label="Step",
            y_label="Loss",
            series=[
                {
                    "label": "train loss",
                    "color": "#0f766e",
                    "points": _extract_points(history, x_key="step", y_key="loss"),
                }
            ],
        ),
        _build_chart(
            title="Head Loss",
            x_label="Step",
            y_label="Loss",
            series=[
                {
                    "label": "head1_loss",
                    "color": "#2563eb",
                    "points": _extract_points(history, x_key="step", y_key="head1_loss"),
                },
                {
                    "label": "head2_loss",
                    "color": "#ea580c",
                    "points": _extract_points(history, x_key="step", y_key="head2_loss"),
                },
                {
                    "label": "head3_loss",
                    "color": "#7c3aed",
                    "points": _extract_points(history, x_key="step", y_key="head3_loss"),
                },
            ],
        ),
        _build_chart(
            title="Head Accuracy",
            x_label="Step",
            y_label="Accuracy",
            series=[
                {
                    "label": "head1_acc",
                    "color": "#16a34a",
                    "points": _extract_points(history, x_key="step", y_key="head1_acc"),
                },
                {
                    "label": "head2_acc",
                    "color": "#f59e0b",
                    "points": _extract_points(history, x_key="step", y_key="head2_acc"),
                },
                {
                    "label": "head3_acc",
                    "color": "#dc2626",
                    "points": _extract_points(history, x_key="step", y_key="head3_acc"),
                },
            ],
        ),
        _build_chart(
            title="Validation Snapshot",
            x_label="Step",
            y_label="Metric",
            series=[
                {
                    "label": "valid loss",
                    "color": "#0f766e",
                    "points": _extract_points(eval_history, x_key="step", y_key="loss"),
                },
                {
                    "label": "valid head1_acc",
                    "color": "#16a34a",
                    "points": _extract_points(eval_history, x_key="step", y_key="head1_acc"),
                },
                {
                    "label": "valid head2_acc",
                    "color": "#f59e0b",
                    "points": _extract_points(eval_history, x_key="step", y_key="head2_acc"),
                },
                {
                    "label": "valid head3_acc",
                    "color": "#dc2626",
                    "points": _extract_points(eval_history, x_key="step", y_key="head3_acc"),
                },
            ],
        ),
    ]

    recent_rows = history[-10:]
    recent_table_rows = "".join(
        "<tr>"
        f"<td>{int(row.get('step', 0))}</td>"
        f"<td>{_safe_float(row.get('loss')) or 0.0:.4f}</td>"
        f"<td>{_safe_float(row.get('head1_loss')) or 0.0:.4f}</td>"
        f"<td>{_safe_float(row.get('head2_loss')) or 0.0:.4f}</td>"
        f"<td>{_safe_float(row.get('head3_loss')) or 0.0:.4f}</td>"
        f"<td>{_safe_float(row.get('head1_acc')) or 0.0:.4f}</td>"
        f"<td>{_safe_float(row.get('head2_acc')) or 0.0:.4f}</td>"
        f"<td>{_safe_float(row.get('head3_acc')) or 0.0:.4f}</td>"
        "</tr>"
        for row in recent_rows
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pythia Medusa Training Dashboard</title>
  <style>
    :root {{
      --bg: #f7f7f2;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --border: #d6d3d1;
      --grid: #e7e5e4;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #fff8eb 0%, var(--bg) 42%, #eef5f4 100%);
      color: var(--ink);
      line-height: 1.5;
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 60px;
    }}
    .hero {{
      background: linear-gradient(135deg, #0f766e 0%, #155e75 48%, #1d4ed8 100%);
      color: white;
      border-radius: 24px;
      padding: 28px 28px 24px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 0;
      opacity: 0.92;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 20px 0 28px;
    }}
    .summary-card, .panel, .chart-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .summary-card {{
      padding: 16px 18px;
    }}
    .summary-label {{
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .summary-value {{
      font-size: 1.4rem;
      font-weight: 700;
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      margin-bottom: 24px;
    }}
    .chart-card {{
      padding: 18px;
    }}
    .chart-card h3, .panel h3 {{
      margin: 0 0 12px;
      font-size: 1.05rem;
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .grid {{
      stroke: var(--grid);
      stroke-width: 1;
    }}
    .axis {{
      stroke: #94a3b8;
      stroke-width: 1.5;
    }}
    .axis-label, .axis-tick {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .panel {{
      padding: 18px;
      margin-bottom: 18px;
    }}
    .config-table, .metrics-table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .config-table th, .config-table td, .metrics-table th, .metrics-table td {{
      text-align: left;
      padding: 8px 10px;
      border-top: 1px solid var(--grid);
      vertical-align: top;
    }}
    .config-table th, .metrics-table th {{
      width: 180px;
      color: var(--muted);
      font-weight: 600;
    }}
    .empty {{
      color: var(--muted);
      margin: 0;
    }}
    @media (max-width: 720px) {{
      .charts {{
        grid-template-columns: 1fr;
      }}
      .hero h1 {{
        font-size: 1.6rem;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Pythia Medusa Training Dashboard</h1>
      <p>本地离线可视化训练过程。重点看总 loss、三个 head 的 loss/acc，以及每轮验证快照。</p>
    </section>

    <section class="summary-grid">
      {_summary_cards(run_payload, summary)}
    </section>

    <section class="charts">
      {''.join(chart_specs)}
    </section>

    <section class="panel">
      <h3>Recent Training Rows</h3>
      <table class="metrics-table">
        <thead>
          <tr>
            <th>Step</th>
            <th>Loss</th>
            <th>Head1 Loss</th>
            <th>Head2 Loss</th>
            <th>Head3 Loss</th>
            <th>Head1 Acc</th>
            <th>Head2 Acc</th>
            <th>Head3 Acc</th>
          </tr>
        </thead>
        <tbody>
          {recent_table_rows}
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h3>Run Config</h3>
      {_config_table(config)}
    </section>
  </div>
</body>
</html>
"""


def write_training_dashboard(
    run_payload: dict[str, Any],
    *,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_training_dashboard_html(run_payload), encoding="utf-8")
    if manifest_path is not None:
        write_json({"dashboard_path": str(output_path)}, manifest_path)
    return output_path


def _load_run_payload(run_dir: Path) -> dict[str, Any]:
    run_summary_path = run_dir / "run_summary.json"
    training_summary_path = run_dir / "training_summary.json"
    if run_summary_path.exists():
        return json.loads(run_summary_path.read_text(encoding="utf-8"))
    if not training_summary_path.exists():
        raise FileNotFoundError(
            f"Could not find run_summary.json or training_summary.json under {run_dir}"
        )
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    return {
        "config": {},
        "parameter_stats": {},
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an offline HTML dashboard for Medusa training.")
    parser.add_argument("--run-dir", required=True, help="Training output directory containing run_summary.json.")
    parser.add_argument("--output", help="Optional HTML output path. Defaults to <run-dir>/training_dashboard.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_payload = _load_run_payload(run_dir)
    output_path = Path(args.output) if args.output else run_dir / "training_dashboard.html"
    path = write_training_dashboard(run_payload, output_path=output_path)
    print(json.dumps({"dashboard_path": str(path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
