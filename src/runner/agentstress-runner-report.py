"""Report generation: JSON and HTML output from stress test results."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def generate_json_report(result: Any, output_path: str | Path) -> Path:
    """Generate a JSON report from a StressTestResult."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "framework": "agentstress",
        "version": "0.1.0",
    }

    if hasattr(result, "to_dict"):
        report_data["result"] = result.to_dict()
    elif isinstance(result, dict):
        report_data["result"] = result
    else:
        report_data["result"] = str(result)

    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    return output_path


def generate_html_report(result: Any, output_path: str | Path) -> Path:
    """Generate an HTML report from a StressTestResult."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = result.to_dict() if hasattr(result, "to_dict") else result

    scenario_name = data.get("scenario_name", "Unknown")
    topology_name = data.get("topology_name", "Unknown")
    baseline_score = data.get("baseline_score", "N/A")
    stressed_score = data.get("stressed_score", "N/A")

    metrics_html = ""
    if "metrics" in data:
        m = data["metrics"]
        quality = m.get("quality", {})
        cost = m.get("cost", {})
        metrics_html = f"""
        <div class="section">
            <h2>Quality Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Baseline Score</td><td>{quality.get('baseline_score', 'N/A')}</td></tr>
                <tr><td>Stressed Score</td><td>{quality.get('stressed_score', 'N/A')}</td></tr>
                <tr><td>Score Delta</td><td>{quality.get('score_delta', 'N/A')}</td></tr>
                <tr><td>Degradation %</td><td>{quality.get('degradation_pct', 'N/A')}%</td></tr>
            </table>
        </div>
        <div class="section">
            <h2>Cost Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Baseline Cost</td><td>${cost.get('baseline_usd', 0):.6f}</td></tr>
                <tr><td>Stressed Cost</td><td>${cost.get('stressed_usd', 0):.6f}</td></tr>
                <tr><td>Overhead Ratio</td><td>{cost.get('overhead_ratio', 'N/A')}x</td></tr>
            </table>
        </div>"""

    criterion_rows = ""
    if "metrics" in data:
        deltas = data["metrics"].get("quality", {}).get("criterion_deltas", {})
        for crit, delta in deltas.items():
            color = "green" if delta >= 0 else "red"
            criterion_rows += f'<tr><td>{crit}</td><td style="color:{color}">{delta:+.4f}</td></tr>'

    criterion_html = ""
    if criterion_rows:
        criterion_html = f"""
        <div class="section">
            <h2>Per-Criterion Impact</h2>
            <table>
                <tr><th>Criterion</th><th>Score Delta</th></tr>
                {criterion_rows}
            </table>
        </div>"""

    curve_html = ""
    if "degradation_curve" in data and data["degradation_curve"]:
        dc = data["degradation_curve"]
        curve_rows = ""
        for p in dc.get("points", []):
            curve_rows += (
                f"<tr><td>{p['fault_probability']}</td>"
                f"<td>{p['quality_score']:.4f}</td>"
                f"<td>${p['cost_usd']:.6f}</td>"
                f"<td>{p['tokens']}</td></tr>"
            )
        curve_html = f"""
        <div class="section">
            <h2>Degradation Curve</h2>
            <p>Resilience Score: <strong>{dc.get('resilience_score', 'N/A')}</strong></p>
            <p>Half-Degradation Point: <strong>{dc.get('half_degradation_point', 'N/A')}</strong></p>
            <table>
                <tr><th>Fault Probability</th><th>Quality</th><th>Cost</th><th>Tokens</th></tr>
                {curve_rows}
            </table>
        </div>"""

    blast_html = ""
    if "blast_radius" in data and data["blast_radius"]:
        br = data["blast_radius"]
        blast_rows = ""
        for ar in br.get("agent_results", []):
            crit_class = {
                "critical": "critical",
                "important": "important",
                "redundant": "redundant",
            }.get(ar["criticality"], "")
            blast_rows += (
                f'<tr class="{crit_class}">'
                f"<td>{ar['agent_id']}</td>"
                f"<td>{ar['agent_role']}</td>"
                f"<td>{ar['degraded_score']:.4f}</td>"
                f"<td>{ar['degradation_pct']:.1f}%</td>"
                f"<td>{ar['criticality']}</td>"
                f"<td>{', '.join(ar['affected_downstream'])}</td></tr>"
            )
        blast_html = f"""
        <div class="section">
            <h2>Blast Radius Analysis</h2>
            <p>Critical Agents: <strong>{', '.join(br.get('critical_agents', [])) or 'None'}</strong></p>
            <p>Redundant Agents: <strong>{', '.join(br.get('redundant_agents', [])) or 'None'}</strong></p>
            <p>System Resilience (avg degradation): <strong>{br.get('system_resilience_avg_degradation_pct', 0):.1f}%</strong></p>
            <table>
                <tr><th>Agent</th><th>Role</th><th>Degraded Score</th><th>Degradation %</th><th>Criticality</th><th>Downstream</th></tr>
                {blast_rows}
            </table>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgentStress Report — {scenario_name}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 960px; margin: 0 auto; padding: 2rem; background: #f8f9fa; color: #212529; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 0.5rem; }}
        h2 {{ color: #16213e; margin-top: 1.5rem; }}
        .header {{ background: #1a1a2e; color: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }}
        .header h1 {{ color: white; border-bottom: none; margin: 0; }}
        .header p {{ margin: 0.25rem 0; opacity: 0.9; }}
        .section {{ background: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
        th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #dee2e6; }}
        th {{ background: #e9ecef; font-weight: 600; }}
        .score-box {{ display: inline-block; padding: 0.5rem 1rem; border-radius: 6px;
                      font-size: 1.25rem; font-weight: bold; margin: 0.25rem; }}
        .score-baseline {{ background: #d4edda; color: #155724; }}
        .score-stressed {{ background: #f8d7da; color: #721c24; }}
        .critical {{ background: #f8d7da; }}
        .important {{ background: #fff3cd; }}
        .redundant {{ background: #d4edda; }}
        .footer {{ text-align: center; color: #6c757d; margin-top: 2rem; font-size: 0.875rem; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AgentStress Report</h1>
        <p>Scenario: {scenario_name} | Topology: {topology_name}</p>
        <p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="section">
        <h2>Overview</h2>
        <span class="score-box score-baseline">Baseline: {baseline_score}</span>
        <span class="score-box score-stressed">Stressed: {stressed_score}</span>
    </div>

    {metrics_html}
    {criterion_html}
    {curve_html}
    {blast_html}

    <div class="footer">
        <p>Generated by AgentStress v0.1.0 — Multi-agent reliability testing framework</p>
    </div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path
