"""AgentStress CLI: command-line interface for stress testing multi-agent systems.

Commands:
    run      — Run a stress test scenario against a topology
    sweep    — Run sweep mode across multiple fault probability levels
    blast    — Run blast radius analysis (kill each agent, measure impact)
    replay   — Replay a recorded stress test run
    viz      — Visualize a topology as Mermaid or Graphviz
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def _load_topology(path: str):
    from topology.agentstress_topology_parse import load_topology_yaml
    return load_topology_yaml(path)


def _load_scenario(path: str):
    from runner.agentstress_runner_scenario import load_scenario_yaml
    return load_scenario_yaml(path)


def cmd_run(args: argparse.Namespace) -> None:
    """Run a stress test scenario."""
    from runner.agentstress_runner_engine import StressTestEngine
    from runner.agentstress_runner_report import generate_html_report, generate_json_report

    topology = _load_topology(args.topology)
    scenario = _load_scenario(args.scenario)

    console.print(f"[bold]AgentStress[/bold] — Running scenario: {scenario.name}")
    console.print(f"  Topology: {topology.name} ({len(topology.agents)} agents)")
    console.print(f"  Faults: {len(scenario.fault_configs)}")

    engine = StressTestEngine(
        topology=topology,
        scenario=scenario,
        model=args.model,
        judge_enabled=not args.no_judge,
    )

    try:
        if args.app_module:
            app = _import_app(args.app_module)
        else:
            console.print("[yellow]No --app-module provided. Using stub graph for demo.[/yellow]")
            app = _create_stub_app(topology)

        inputs = json.loads(args.inputs) if args.inputs else {}
        result = asyncio.run(engine.run(app, inputs))

        output_dir = Path(args.output_dir)
        json_path = generate_json_report(result, output_dir / "report.json")
        html_path = generate_html_report(result, output_dir / "report.html")

        console.print(f"\n[green]Done![/green]")
        if result.metrics:
            _print_metrics_table(result.metrics)
        console.print(f"\nReports: {json_path}, {html_path}")

    finally:
        engine.close()


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run sweep mode across fault probability levels."""
    from runner.agentstress_runner_engine import StressTestEngine
    from runner.agentstress_runner_report import generate_html_report, generate_json_report

    topology = _load_topology(args.topology)
    scenario = _load_scenario(args.scenario)

    if not scenario.sweep.enabled:
        console.print("[red]Sweep not enabled in this scenario config.[/red]")
        return

    console.print(f"[bold]AgentStress Sweep[/bold] — {scenario.name}")
    console.print(f"  Probabilities: {scenario.sweep.values}")

    engine = StressTestEngine(
        topology=topology,
        scenario=scenario,
        model=args.model,
        judge_enabled=not args.no_judge,
    )

    try:
        app = _import_app(args.app_module) if args.app_module else _create_stub_app(topology)
        inputs = json.loads(args.inputs) if args.inputs else {}
        result = asyncio.run(engine.run_sweep(app, inputs))

        output_dir = Path(args.output_dir)
        generate_json_report(result, output_dir / "sweep_report.json")
        generate_html_report(result, output_dir / "sweep_report.html")

        if result.degradation_curve:
            _print_curve_table(result.degradation_curve)

        console.print(f"\n[green]Sweep complete![/green]")

    finally:
        engine.close()


def cmd_blast(args: argparse.Namespace) -> None:
    """Run blast radius analysis."""
    from runner.agentstress_runner_engine import StressTestEngine
    from runner.agentstress_runner_report import generate_html_report, generate_json_report

    topology = _load_topology(args.topology)
    scenario = _load_scenario(args.scenario) if args.scenario else None

    if scenario is None:
        from runner.agentstress_runner_scenario import ScenarioSpec
        scenario = ScenarioSpec(name="blast-radius")

    console.print(f"[bold]AgentStress Blast Radius[/bold] — {topology.name}")
    console.print(f"  Agents: {', '.join(topology.get_agent_ids())}")

    engine = StressTestEngine(
        topology=topology,
        scenario=scenario,
        model=args.model,
        judge_enabled=not args.no_judge,
    )

    try:
        app = _import_app(args.app_module) if args.app_module else _create_stub_app(topology)
        inputs = json.loads(args.inputs) if args.inputs else {}
        result = asyncio.run(engine.run_blast_radius(app, inputs))

        output_dir = Path(args.output_dir)
        generate_json_report(result, output_dir / "blast_report.json")
        generate_html_report(result, output_dir / "blast_report.html")

        if result.blast_radius:
            _print_blast_table(result.blast_radius)

        console.print(f"\n[green]Blast radius analysis complete![/green]")

    finally:
        engine.close()


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay a recorded stress test run."""
    from replay.agentstress_replay_record import ExecutionRecorder
    from replay.agentstress_replay_player import ReplayPlayer

    console.print(f"[bold]AgentStress Replay[/bold] — Loading: {args.recording}")

    recording = ExecutionRecorder.load(args.recording)
    player = ReplayPlayer(recording)

    console.print(f"  Run ID: {recording.run_id}")
    console.print(f"  Steps: {player.total_steps}")
    console.print(f"  Agents: {', '.join(set(s.agent_id for s in recording.steps))}")

    if args.step is not None:
        step = player.jump_to(args.step)
        if step:
            _print_step_detail(step, args.step)
        else:
            console.print(f"[red]Step {args.step} out of range[/red]")
    else:
        table = Table(title="Recorded Steps")
        table.add_column("Index", style="dim")
        table.add_column("Agent")
        table.add_column("Fault")
        table.add_column("Output (truncated)")

        for i, step in enumerate(recording.steps):
            fault_str = step.fault_applied or "-"
            output = step.output_content[:60] + "..." if len(step.output_content) > 60 else step.output_content
            table.add_row(str(i), step.agent_id, fault_str, output)

        console.print(table)


def cmd_viz(args: argparse.Namespace) -> None:
    """Visualize a topology."""
    from topology.agentstress_topology_visualize import (
        save_graphviz,
        save_mermaid,
        to_graphviz,
        to_mermaid,
    )

    topology = _load_topology(args.topology)

    if args.format == "mermaid":
        if args.output:
            save_mermaid(topology, args.output, args.direction)
            console.print(f"[green]Mermaid saved to {args.output}[/green]")
        else:
            console.print(to_mermaid(topology, args.direction))
    elif args.format == "dot":
        if args.output:
            save_graphviz(topology, args.output, args.direction)
            console.print(f"[green]DOT saved to {args.output}[/green]")
        else:
            console.print(to_graphviz(topology, args.direction))


# --- Helpers ---

def _import_app(module_path: str):
    """Import a LangGraph app from a module path like 'my.module:app'."""
    if ":" not in module_path:
        raise ValueError("App module must be in format 'module.path:attribute'")
    mod_path, attr_name = module_path.rsplit(":", 1)

    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr_name)


def _create_stub_app(topology):
    """Create a minimal stub LangGraph app that calls LLMs through the pipeline."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any

    class _StubLLM(BaseChatModel):
        agent_name: str = "stub"

        @property
        def _llm_type(self) -> str:
            return "stub"

        def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
            content = f"[{self.agent_name}] Processed {len(messages)} messages."
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    class _StubNode:
        def __init__(self, name: str):
            self.llm = _StubLLM(agent_name=name)

    class StubApp:
        def __init__(self, agent_ids):
            self._agents = {aid: _StubNode(aid) for aid in agent_ids}
            self.nodes = self._agents

        async def ainvoke(self, inputs):
            messages = [HumanMessage(content=str(inputs))]
            last_output = ""
            for aid in self._agents:
                result = self._agents[aid].llm.invoke(messages)
                messages.append(result)
                last_output = result.content
            return {"output": last_output, **inputs}

        def invoke(self, inputs):
            import asyncio
            return asyncio.get_event_loop().run_until_complete(self.ainvoke(inputs))

    return StubApp(topology.get_agent_ids())


def _print_metrics_table(metrics) -> None:
    table = Table(title="Stress Test Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Baseline Score", f"{metrics.baseline_score:.4f}")
    table.add_row("Stressed Score", f"{metrics.stressed_score:.4f}")
    table.add_row("Degradation", f"{metrics.degradation_pct:.1f}%")
    table.add_row("Cost Overhead", f"{metrics.cost_overhead_ratio:.2f}x")
    table.add_row("Faults Triggered", str(metrics.faults_triggered))

    console.print(table)


def _print_curve_table(curve) -> None:
    table = Table(title="Degradation Curve")
    table.add_column("Fault Prob")
    table.add_column("Quality")
    table.add_column("Cost ($)")
    table.add_column("Tokens")

    for p in curve.points:
        table.add_row(
            f"{p.fault_probability:.1%}",
            f"{p.quality_score:.4f}",
            f"${p.cost_usd:.6f}",
            str(p.tokens),
        )

    console.print(table)
    console.print(f"  Resilience Score: {curve.resilience_score():.4f}")
    hdp = curve.half_degradation_point()
    if hdp:
        console.print(f"  Half-Degradation Point: {hdp:.1%}")


def _print_blast_table(report) -> None:
    table = Table(title="Blast Radius Analysis")
    table.add_column("Agent")
    table.add_column("Role")
    table.add_column("Degradation")
    table.add_column("Criticality")
    table.add_column("Downstream")

    for r in report.agent_results:
        style = {"critical": "red bold", "important": "yellow", "redundant": "green"}.get(
            r.criticality.value, ""
        )
        table.add_row(
            r.agent_id,
            r.agent_role,
            f"{r.degradation_pct:.1f}%",
            r.criticality.value,
            ", ".join(r.affected_downstream) or "-",
            style=style,
        )

    console.print(table)
    console.print(f"  Critical: {', '.join(report.critical_agents) or 'None'}")
    console.print(f"  Redundant: {', '.join(report.redundant_agents) or 'None'}")


def _print_step_detail(step, index: int) -> None:
    console.print(f"\n[bold]Step {index}[/bold] — Agent: {step.agent_id}")
    console.print(f"  Fault: {step.fault_applied or 'None'}")
    console.print(f"  Input messages: {len(step.input_messages)}")
    console.print(f"  Output:")
    console.print(f"    {step.output_content[:500]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentstress",
        description="AgentStress — Multi-agent reliability testing framework",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--topology", "-t", required=True, help="Topology YAML path")
    common.add_argument("--model", default="default", help="Model identifier for cost tracking")
    common.add_argument("--no-judge", action="store_true", help="Disable judge evaluation")
    common.add_argument("--app-module", help="Python module:attr for the LangGraph app")
    common.add_argument("--inputs", help="JSON string of inputs to the app")
    common.add_argument("--output-dir", "-o", default="result", help="Output directory")

    # run
    p_run = subparsers.add_parser("run", parents=[common], help="Run a stress test")
    p_run.add_argument("--scenario", "-s", required=True, help="Scenario YAML path")

    # sweep
    p_sweep = subparsers.add_parser("sweep", parents=[common], help="Run sweep mode")
    p_sweep.add_argument("--scenario", "-s", required=True, help="Scenario YAML path")

    # blast
    p_blast = subparsers.add_parser("blast", parents=[common], help="Run blast radius analysis")
    p_blast.add_argument("--scenario", "-s", help="Optional scenario YAML")

    # replay
    p_replay = subparsers.add_parser("replay", help="Replay a recorded run")
    p_replay.add_argument("recording", help="Path to recording JSON file")
    p_replay.add_argument("--step", type=int, help="Jump to a specific step")

    # viz
    p_viz = subparsers.add_parser("viz", help="Visualize a topology")
    p_viz.add_argument("--topology", "-t", required=True, help="Topology YAML path")
    p_viz.add_argument("--format", choices=["mermaid", "dot"], default="mermaid")
    p_viz.add_argument("--output", "-o", help="Output file path")
    p_viz.add_argument("--direction", default="LR", choices=["LR", "TD", "TB"])

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Ensure src/ is on path
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    commands = {
        "run": cmd_run,
        "sweep": cmd_sweep,
        "blast": cmd_blast,
        "replay": cmd_replay,
        "viz": cmd_viz,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
