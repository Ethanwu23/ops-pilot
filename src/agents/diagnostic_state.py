from typing import Literal

from langgraph.graph import MessagesState

from schema.diagnosis import AnalysisResult, Evidence


class DiagnosticState(MessagesState, total=False):
    """保存诊断工作流中的消息、执行状态和错误信息。"""

    status: Literal["succeeded", "failed", "running"]
    error: str | None
    evidence: list[Evidence]
    analysis: AnalysisResult | None


def start_diagnosis(state: DiagnosticState) -> DiagnosticState:
    """开始本轮诊断，重置执行状态和错误信息。"""
    return {
        "status": "running",
        "error": None,
        "evidence": [],
        "analysis": None,
    }
