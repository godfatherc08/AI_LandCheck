"""Dataclasses and type definitions for verification agent"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class Signal:
    """A single verification signal from a tool"""
    source: str
    claim_checked: str
    result: str
    weight: float
    severity: str
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """A single reasoning step in the agent's process"""
    step: int
    tool_called: str
    inputs: Dict[str, Any]
    output: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VerificationResult:
    """Final verification result from agent"""
    overall_risk: str                               
    signals: List[Signal]
    reasoning_trace: List[ReasoningStep]
    recommendation: str
    squad_action: Optional[str] = None
    trust_score: Optional[int] = None