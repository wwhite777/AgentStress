"""MedAgent demo: stress test the MedAgent clinical pipeline.

Demonstrates AgentStress against the MedAgent LangGraph application,
injecting context corruption + hallucination faults and evaluating
with the local judge model.

Requires:
    - MedAgent graph at /home/wjeong/ma/medagent/
    - vLLM judge server running (scripts/agentstress-serve-judge.sh)
    - API keys for LLM providers (or local vLLM serving agents)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import topology  # noqa: F401
import proxy  # noqa: F401
import faults  # noqa: F401
import telemetry  # noqa: F401
import adapters  # noqa: F401
import eval  # noqa: F401
import runner  # noqa: F401
import replay  # noqa: F401

from runner.agentstress_runner_engine import StressTestEngine
from runner.agentstress_runner_report import generate_html_report, generate_json_report
from runner.agentstress_runner_scenario import load_scenario_yaml
from topology.agentstress_topology_parse import load_topology_yaml
from topology.agentstress_topology_visualize import save_mermaid


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULT_DIR = Path(__file__).resolve().parent.parent / "result"

MEDAGENT_DIR = Path("/home/wjeong/ma/medagent")

DEMO_INPUT = {
    "patient_input": (
        "45-year-old male presenting to the emergency department with acute onset "
        "chest pain radiating to the left arm, associated with diaphoresis and "
        "shortness of breath. History of hypertension and hyperlipidemia. "
        "Vitals: HR 110, BP 150/95, SpO2 94%, Temp 37.2C."
    ),
    "messages": [],
}


def _load_medagent_app():
    """Attempt to load the MedAgent LangGraph app."""
    if not MEDAGENT_DIR.exists():
        return None

    # MedAgent's internal imports use "src.graph.medagent-..." style,
    # so we need the project root (parent of src/) on sys.path.
    medagent_root = str(MEDAGENT_DIR)
    if medagent_root not in sys.path:
        sys.path.insert(0, medagent_root)

    try:
        import importlib.util

        graph_file = MEDAGENT_DIR / "src" / "graph" / "medagent-clinical-graph.py"
        spec = importlib.util.spec_from_file_location(
            "medagent_clinical_graph", graph_file,
        )
        graph_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(graph_mod)

        if hasattr(graph_mod, "build_default_app"):
            return graph_mod.build_default_app()
        elif hasattr(graph_mod, "app"):
            return graph_mod.app
    except Exception as e:
        print(f"[WARN] Could not load MedAgent app: {e}")

    return None


def _create_stub_medagent():
    """Create a stub MedAgent app that invokes LLMs through the pipeline.

    Each node's LLM is a BaseChatModel, so LangGraphAdapter.wrap() replaces
    them with ProxiedChatModels. The ainvoke() method calls each LLM in
    sequence so the InterceptionPipeline actually processes every call.
    """
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from typing import Any

    class StubMedAgent(BaseChatModel):
        agent_name: str = "stub"

        @property
        def _llm_type(self) -> str:
            return "stub-medagent"

        def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
            # Count non-system messages to reflect context availability.
            # When context truncation is applied, fewer messages arrive,
            # producing visibly degraded output.
            context_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
            context_depth = len(context_msgs)

            responses = {
                "triage": (
                    "ESI Level 2. Chief complaint: acute chest pain radiating to left arm. "
                    "Suspected Acute Coronary Syndrome. Vitals concerning for hemodynamic instability. "
                    "Recommend immediate cardiac workup."
                ),
                "literature": (
                    "[Retrieved Context]\n"
                    "ACS guidelines (2025 AHA/ACC): Troponin testing within 1hr of presentation. "
                    "12-lead ECG within 10 minutes. Aspirin 325mg PO stat. "
                    "Risk stratification via HEART score recommended for chest pain patients. "
                    "Literature supports early invasive strategy for NSTEMI with HEART >= 7."
                ),
                "reasoning": (
                    "Clinical Reasoning Analysis:\n"
                    "Differential Diagnosis:\n"
                    "  1. STEMI — probability 0.55 (ST changes + troponin + classic presentation)\n"
                    "  2. NSTEMI — probability 0.30 (troponin elevation without ST elevation)\n"
                    "  3. Pulmonary Embolism — probability 0.10 (dyspnea, tachycardia)\n"
                    "  4. Aortic Dissection — probability 0.05 (radiating pain, hypertension)\n"
                    "Recommended workup: Serial troponins, 12-lead ECG, chest X-ray, D-dimer if PE suspected.\n"
                    "Drug interactions: None with current medications.\n"
                    "Lab interpretation: Elevated troponin I at 2.5 ng/mL (critical)."
                ) if context_depth >= 3 else (
                    "Clinical Reasoning Analysis:\n"
                    "Differential Diagnosis:\n"
                    "  1. Chest pain — probability 0.50 (insufficient context for specific diagnosis)\n"
                    "  2. Cardiac event — probability 0.30 (based on limited available data)\n"
                    "[WARNING: Reasoning degraded due to incomplete upstream context. "
                    "Missing triage details and/or literature evidence.]"
                ),
                "report": self._generate_report(context_msgs),
            }
            return ChatResult(
                generations=[ChatGeneration(
                    message=AIMessage(content=responses.get(self.agent_name, "Stub response."))
                )]
            )

        def _generate_report(self, context_msgs: list) -> str:
            """Report quality depends on upstream content quality."""
            all_text = " ".join(
                m.content for m in context_msgs if isinstance(m.content, str)
            )
            # Debug: uncomment to trace report input
            # if self.agent_name == "report":
            #     print(f"  [DEBUG] {len(context_msgs)} msgs, STEMI={'STEMI' in all_text}")
            has_differential = "STEMI" in all_text and "NSTEMI" in all_text
            has_literature = "ACS guidelines" in all_text or "AHA" in all_text
            has_triage = "ESI" in all_text

            if has_differential and has_literature and has_triage:
                return (
                    "SOAP Note:\n"
                    "S: 45-year-old male presenting with acute onset chest pain radiating to left arm, "
                    "associated with diaphoresis and shortness of breath. History of HTN, HLD.\n"
                    "O: HR 110, BP 150/95, SpO2 94%, Temp 37.2C. Troponin I 2.5 ng/mL (elevated). "
                    "ECG pending.\n"
                    "A: Acute Coronary Syndrome, most likely STEMI. High-risk per HEART score.\n"
                    "P: 1) Aspirin 325mg PO stat 2) Serial troponins q3h 3) 12-lead ECG STAT "
                    "4) Cardiology consult for emergent cath 5) Telemetry 6) IV x2, type+screen"
                )
            elif has_differential or has_triage:
                return (
                    "SOAP Note:\n"
                    "S: 45M with chest pain. Limited upstream data available.\n"
                    "O: Vitals partially documented. Labs pending full review.\n"
                    "A: Suspected cardiac event. Differential incomplete — missing literature context.\n"
                    "P: Basic cardiac workup. Further evaluation needed once upstream data available. "
                    "[DEGRADED: Missing literature or triage context from upstream agents]"
                )
            else:
                return (
                    "SOAP Note:\n"
                    "S: Patient with chest pain (details unavailable).\n"
                    "O: Insufficient objective data.\n"
                    "A: Cannot form assessment — upstream agents failed to provide required data.\n"
                    "P: Emergency stabilization only. Full workup blocked by system failure. "
                    "[CRITICAL DEGRADATION: All upstream context lost due to fault cascade]"
                )

    class StubNode:
        """A node that holds an LLM — LangGraphAdapter.wrap() finds and proxies it."""
        def __init__(self, agent_name: str):
            self.llm = StubMedAgent(agent_name=agent_name)

    class StubApp:
        """Stub LangGraph app that calls each node's LLM through the pipeline."""
        def __init__(self):
            self._node_objs = {
                aid: StubNode(aid)
                for aid in ["triage", "literature", "reasoning", "report"]
            }
            self.nodes = self._node_objs

        async def ainvoke(self, inputs: dict) -> dict:
            patient_input = inputs.get("patient_input", "No patient data provided.")

            # Call each agent's LLM in sequence — these go through the
            # InterceptionPipeline (faults + telemetry) because wrap() replaced
            # each node's .llm with a ProxiedChatModel.
            messages = [
                SystemMessage(content="You are a clinical AI agent in a multi-agent medical system."),
                HumanMessage(content=patient_input),
            ]

            triage_result = self._node_objs["triage"].llm.invoke(messages)
            messages.append(triage_result)

            lit_messages = messages + [
                HumanMessage(content="Search for relevant clinical guidelines and literature."),
            ]
            lit_result = self._node_objs["literature"].llm.invoke(lit_messages)
            messages.append(lit_result)

            reason_messages = messages + [
                HumanMessage(content="Provide differential diagnosis and recommended workup."),
            ]
            reason_result = self._node_objs["reasoning"].llm.invoke(reason_messages)
            messages.append(reason_result)

            report_messages = messages + [
                HumanMessage(content="Generate a complete SOAP note."),
            ]
            report_result = self._node_objs["report"].llm.invoke(report_messages)

            return {
                "soap_note": report_result.content,
                "triage_output": triage_result.content,
                "literature_output": lit_result.content,
                "reasoning_output": reason_result.content,
            }

    return StubApp()


async def run_demo():
    print("=" * 60)
    print("AgentStress — MedAgent Clinical Pipeline Demo")
    print("=" * 60)

    # Load topology + scenario
    topology = load_topology_yaml(CONFIGS_DIR / "agentstress-topology-example.yaml")
    scenario = load_scenario_yaml(CONFIGS_DIR / "agentstress-medagent-demo.yaml")

    print(f"\nTopology: {topology.name} ({len(topology.agents)} agents)")
    print(f"Scenario: {scenario.name} ({len(scenario.fault_configs)} faults)")

    # Save topology visualization
    viz_path = RESULT_DIR / "figure" / "medagent-topology.md"
    save_mermaid(topology, viz_path)
    print(f"Topology viz: {viz_path}")

    # Load or stub the app
    app = _load_medagent_app()
    if app is None:
        print("\n[INFO] MedAgent app not available — using stub for demo.")
        app = _create_stub_medagent()
    else:
        print("\n[INFO] Loaded MedAgent LangGraph app.")

    # Run stress test
    engine = StressTestEngine(
        topology=topology,
        scenario=scenario,
        model="local-vllm",
        judge_enabled=True,  # vLLM judge server running on localhost:8000
    )

    try:
        print("\nRunning stress test...")
        result = await engine.run(app, DEMO_INPUT)

        # Reports
        output_dir = RESULT_DIR / "medagent-demo"
        json_path = generate_json_report(result, output_dir / "report.json")
        html_path = generate_html_report(result, output_dir / "report.html")

        print(f"\nJSON report: {json_path}")
        print(f"HTML report: {html_path}")

        # Print summary
        # Print comparison
        print(f"\n--- Baseline Output (report agent) ---")
        print(result.baseline.output_text[:400])
        print(f"\n--- Stressed Output (report agent) ---")
        print(result.stressed.output_text[:400])

        if result.comparison:
            c = result.comparison
            print(f"\n--- Comparison ---")
            print(f"  Quality delta: {c.quality_delta}")
            print(f"  Total faults triggered: {c.total_faults_triggered}")
            print(f"  Baseline tokens: {c.baseline_total_tokens}")
            print(f"  Stressed tokens: {c.stressed_total_tokens}")
            outputs_differ = result.baseline.output_text != result.stressed.output_text
            print(f"  Outputs differ: {outputs_differ}")
            if c.agent_comparisons:
                print(f"\n  Per-agent telemetry:")
                for ac in c.agent_comparisons:
                    print(f"    {ac.agent_id}: baseline={ac.baseline_tokens}tok "
                          f"stressed={ac.stressed_tokens}tok faults={ac.faults_received}")

        # Run sweep if enabled
        if scenario.sweep.enabled:
            print(f"\nRunning sweep ({scenario.sweep.values})...")
            sweep_result = await engine.run_sweep(app, DEMO_INPUT)
            generate_json_report(sweep_result, output_dir / "sweep_report.json")
            generate_html_report(sweep_result, output_dir / "sweep_report.html")

            if sweep_result.degradation_curve:
                dc = sweep_result.degradation_curve
                print(f"  Resilience: {dc.resilience_score():.4f}")
                hdp = dc.half_degradation_point()
                if hdp:
                    print(f"  Half-degradation at: {hdp:.0%}")

    finally:
        engine.close()

    print("\n" + "=" * 60)
    print("Demo complete!")


def main():
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
