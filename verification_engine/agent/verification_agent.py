"""Public entry point for verification agent"""

from typing import Dict, Optional
from verification_engine.agent.agent import VerificationAgent
from verification_engine.agent.schema import VerificationResult


def verify_document(claims: Dict,
                    plausibility_results: Optional[Dict] = None,
                    vision_flags: Optional[Dict] = None) -> Dict:
    """
    Verify a document using AI agent with external tools

    Args:
        claims: Extracted document fields from Stage 2
        plausibility_results: Results from Stage 3 (optional)
        vision_flags: Vision forensic flags (optional)

    Returns:
        Dictionary with verification results
    """
    agent = VerificationAgent()
    result = agent.run(claims, plausibility_results, vision_flags)

                                                  
    return {
        "overall_risk": result.overall_risk,
        "signals": [
            {
                "source": s.source,
                "claim_checked": s.claim_checked,
                "result": s.result,
                "weight": s.weight,
                "severity": s.severity,
                "explanation": s.explanation,
            }
            for s in result.signals
        ],
        "reasoning_trace": [
            {
                "step": r.step,
                "tool_called": r.tool_called,
                "inputs": r.inputs,
                "output": r.output,
            }
            for r in result.reasoning_trace
        ],
        "recommendation": result.recommendation,
        "squad_action": result.squad_action,
        "trust_score": result.trust_score,
    }


                                               
def verify_document_compat(claims: Dict, plausibility_results: Dict = None) -> Dict:
    """Compatibility wrapper for existing code"""
    return verify_document(claims, plausibility_results)