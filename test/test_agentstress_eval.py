"""Tests for evaluation modules (judge, score, blast, compare)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from eval.agentstress_eval_judge import JudgeResult, LocalJudge
from eval.agentstress_eval_score import (
    DegradationCurve,
    DegradationPoint,
    StressTestMetrics,
    compute_metrics,
)
from eval.agentstress_eval_blast import (
    AgentBlastResult,
    AgentCriticality,
    BlastRadiusAnalyzer,
    BlastRadiusReport,
)
from eval.agentstress_eval_compare import (
    AgentComparison,
    ComparisonReport,
    compare_runs,
)
from telemetry.agentstress_telemetry_collect import StepMetrics, TelemetryCollector
from telemetry.agentstress_telemetry_cost import CostTracker


# --- JudgeResult ---

class TestAgentStressJudgeResult:

    def test_agentstress_judge_result_passed(self):
        r = JudgeResult(overall_score=0.8, scores={"quality": 0.8})
        assert r.passed

    def test_agentstress_judge_result_failed_low_score(self):
        r = JudgeResult(overall_score=0.3)
        assert not r.passed

    def test_agentstress_judge_result_failed_error(self):
        r = JudgeResult(overall_score=0.9, error="Connection failed")
        assert not r.passed


class TestAgentStressLocalJudge:

    def test_agentstress_judge_build_prompt(self):
        judge = LocalJudge(base_url="http://localhost:8000/v1")
        prompt = judge._build_prompt("Agent output text", "Expected output")
        assert "Agent output text" in prompt
        assert "Expected output" in prompt
        assert "task_completion" in prompt

    def test_agentstress_judge_parse_valid_json(self):
        judge = LocalJudge()
        raw = json.dumps({
            "scores": {"task_completion": 0.9, "output_quality": 0.85},
            "rationales": {"task_completion": "Good", "output_quality": "Solid"},
            "overall_score": 0.875,
        })
        result = judge._parse_response(raw)
        assert result.overall_score == 0.875
        assert result.scores["task_completion"] == 0.9
        assert result.error is None

    def test_agentstress_judge_parse_no_overall_computes_avg(self):
        judge = LocalJudge()
        raw = json.dumps({
            "scores": {"a": 0.8, "b": 0.6},
            "rationales": {},
        })
        result = judge._parse_response(raw)
        assert result.overall_score == pytest.approx(0.7)

    def test_agentstress_judge_parse_invalid_json(self):
        judge = LocalJudge()
        result = judge._parse_response("Not JSON at all")
        assert result.error is not None

    def test_agentstress_judge_parse_json_in_text(self):
        judge = LocalJudge()
        raw = 'Here is my evaluation:\n{"scores": {"quality": 0.7}, "overall_score": 0.7}\nDone.'
        result = judge._parse_response(raw)
        assert result.overall_score == 0.7

    def test_agentstress_judge_evaluate_connection_error(self):
        judge = LocalJudge(base_url="http://localhost:99999/v1")
        result = judge.evaluate("Test output")
        assert result.error is not None
        assert "Cannot connect" in result.error


# --- StressTestMetrics ---

class TestAgentStressMetrics:

    def test_agentstress_metrics_to_dict(self):
        m = StressTestMetrics(
            scenario_name="test",
            fault_type="context_truncation",
            fault_probability=0.5,
            baseline_score=0.9,
            stressed_score=0.6,
            score_delta=-0.3,
            degradation_pct=33.33,
        )
        d = m.to_dict()
        assert d["quality"]["baseline_score"] == 0.9
        assert d["quality"]["degradation_pct"] == 33.33

    def test_agentstress_metrics_compute(self):
        baseline_judge = JudgeResult(
            overall_score=0.9,
            scores={"quality": 0.9, "safety": 0.95},
        )
        stressed_judge = JudgeResult(
            overall_score=0.6,
            scores={"quality": 0.55, "safety": 0.7},
        )
        m = compute_metrics(
            scenario_name="test",
            fault_type="context_truncation",
            fault_probability=0.5,
            baseline_judge_result=baseline_judge,
            stressed_judge_result=stressed_judge,
        )
        assert m.baseline_score == 0.9
        assert m.stressed_score == 0.6
        assert m.score_delta == pytest.approx(-0.3)
        assert m.degradation_pct > 0
        assert "quality" in m.criterion_deltas


# --- DegradationCurve ---

class TestAgentStressDegradationCurve:

    @pytest.fixture
    def agentstress_curve(self):
        curve = DegradationCurve(
            scenario_name="test",
            fault_type="context_truncation",
            baseline_score=0.9,
            baseline_cost_usd=0.01,
        )
        curve.add_point(0.1, 0.88, 0.011, 1100, 50.0)
        curve.add_point(0.3, 0.75, 0.013, 1300, 60.0)
        curve.add_point(0.5, 0.60, 0.015, 1500, 70.0)
        curve.add_point(0.7, 0.40, 0.018, 1800, 80.0)
        curve.add_point(1.0, 0.20, 0.025, 2500, 100.0)
        return curve

    def test_agentstress_curve_sorted(self, agentstress_curve):
        probs = agentstress_curve.probabilities
        assert probs == sorted(probs)

    def test_agentstress_curve_quality_at(self, agentstress_curve):
        assert agentstress_curve.quality_at(0.5) == 0.60
        assert agentstress_curve.quality_at(0.99) is None

    def test_agentstress_curve_half_degradation(self, agentstress_curve):
        # baseline=0.9, half=0.45. First point <=0.45 is at prob=0.7 (score=0.40)
        hdp = agentstress_curve.half_degradation_point()
        assert hdp == 0.7

    def test_agentstress_curve_resilience_score(self, agentstress_curve):
        rs = agentstress_curve.resilience_score()
        assert 0 < rs < 1

    def test_agentstress_curve_to_dict(self, agentstress_curve):
        d = agentstress_curve.to_dict()
        assert len(d["points"]) == 5
        assert "resilience_score" in d

    def test_agentstress_curve_empty(self):
        curve = DegradationCurve("empty", "none", 0.0, 0.0)
        assert curve.resilience_score() == 0.0
        assert curve.half_degradation_point() is None


# --- BlastRadiusAnalyzer ---

class TestAgentStressBlastRadius:

    def test_agentstress_blast_classify_critical(self):
        analyzer = BlastRadiusAnalyzer()
        assert analyzer.classify_criticality(60.0) == AgentCriticality.CRITICAL

    def test_agentstress_blast_classify_important(self):
        analyzer = BlastRadiusAnalyzer()
        assert analyzer.classify_criticality(35.0) == AgentCriticality.IMPORTANT

    def test_agentstress_blast_classify_redundant(self):
        analyzer = BlastRadiusAnalyzer()
        assert analyzer.classify_criticality(10.0) == AgentCriticality.REDUNDANT

    def test_agentstress_blast_analyze_agent(self):
        analyzer = BlastRadiusAnalyzer()
        result = analyzer.analyze_agent(
            agent_id="reasoning",
            agent_role="worker",
            baseline_score=0.9,
            degraded_score=0.3,
            downstream_agents=["report"],
        )
        assert result.degradation_pct == pytest.approx(66.67)
        assert result.criticality == AgentCriticality.CRITICAL
        assert result.affected_downstream == ["report"]

    def test_agentstress_blast_report(self):
        analyzer = BlastRadiusAnalyzer()
        results = [
            analyzer.analyze_agent("triage", "router", 0.9, 0.2, ["reasoning", "literature"]),
            analyzer.analyze_agent("literature", "worker", 0.9, 0.7, ["reasoning"]),
            analyzer.analyze_agent("reasoning", "worker", 0.9, 0.3, ["report"]),
            analyzer.analyze_agent("report", "aggregator", 0.9, 0.8, []),
        ]
        report = analyzer.build_report("medagent", 0.9, results)

        assert "triage" in report.critical_agents
        assert "reasoning" in report.critical_agents
        assert report.most_critical_agent == "triage"
        assert len(report.redundant_agents) > 0
        assert report.system_resilience > 0

    def test_agentstress_blast_report_to_dict(self):
        analyzer = BlastRadiusAnalyzer()
        results = [analyzer.analyze_agent("a", "worker", 0.9, 0.5, [])]
        report = analyzer.build_report("test", 0.9, results)
        d = report.to_dict()
        assert "agent_results" in d
        assert "most_critical_agent" in d


# --- ComparisonReport ---

class TestAgentStressCompare:

    def _make_telemetry(self, agent_steps: dict[str, list[tuple[int, int]]]) -> TelemetryCollector:
        collector = TelemetryCollector()
        for agent_id, steps in agent_steps.items():
            for inp, out in steps:
                ctx_before = MagicMock()
                ctx_before.agent_id = agent_id
                ctx_before.step_index = 1
                ctx_before.messages = [HumanMessage(content="x" * (inp * 4))]
                ctx_before.metadata = {}
                ctx_before.timestamp = 0.0

                collector.before_call(ctx_before)

                ctx_after = MagicMock()
                ctx_after.agent_id = agent_id
                ctx_after.step_index = 1
                ctx_after.response = AIMessage(content="y" * (out * 4))
                ctx_after.metadata = ctx_before.metadata
                ctx_after.fault_applied = None
                ctx_after.timestamp = 0.0

                collector.after_call(ctx_after)

        return collector

    def test_agentstress_compare_basic(self):
        baseline_tel = self._make_telemetry({"triage": [(100, 50)], "reasoning": [(200, 100)]})
        stressed_tel = self._make_telemetry({"triage": [(100, 50)], "reasoning": [(400, 200)]})

        report = compare_runs(
            scenario_name="test",
            baseline_score=0.9,
            stressed_score=0.6,
            baseline_telemetry=baseline_tel,
            stressed_telemetry=stressed_tel,
        )

        assert report.quality_delta == pytest.approx(-0.3)
        assert report.degradation_pct > 0
        assert len(report.agent_comparisons) == 2

    def test_agentstress_compare_to_dict(self):
        baseline_tel = self._make_telemetry({"a": [(100, 50)]})
        stressed_tel = self._make_telemetry({"a": [(200, 100)]})

        report = compare_runs("test", 0.9, 0.7, baseline_tel, stressed_tel)
        d = report.to_dict()
        assert "quality" in d
        assert "cost" in d
        assert "agent_comparisons" in d
