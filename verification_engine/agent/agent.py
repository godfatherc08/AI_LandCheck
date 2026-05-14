"""Main verification agent runner"""

import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

from verification_engine.agent.config import MAX_AGENT_STEPS
from verification_engine.agent.llm import get_llm_client, get_llm_model
from verification_engine.agent.tools import TOOL_FUNCTIONS
from verification_engine.agent.prompts import SYSTEM_PROMPT, build_tool_schemas
from verification_engine.agent.schema import Signal, ReasoningStep, VerificationResult


class VerificationAgent:
    """AI-powered verification agent using Groq Llama 3.3 70B"""

    def __init__(self, max_steps: int = MAX_AGENT_STEPS):
        self.max_steps = max_steps
        self.client = get_llm_client()
        self.model = get_llm_model()
        self.tools = build_tool_schemas()

    def _clean_claims(self, claims: Dict) -> Dict:
        """Remove null/empty values from claims to prevent tool call errors"""
        cleaned = {}

        for key, value in claims.items():
            if value is None:
                continue
            if isinstance(value, dict):
                                                
                nested = self._clean_claims(value)
                if nested:                          
                    cleaned[key] = nested
            elif isinstance(value, list):
                                                  
                filtered = [v for v in value if v not in [None, "", []]]
                if filtered:
                    cleaned[key] = filtered
            elif value not in [None, "", [], {}]:
                cleaned[key] = value

        return cleaned

    def _clean_owner_name(self, name: str) -> str:
        """Clean OCR errors from owner names"""
        if not name:
            return ""

                               
        import re
        cleaned = name.upper()

                                   
        corrections = {
            r'QLADIPUPO': 'OLADIPUPO',
            r'ALABI\s+OFF': 'ALABI, OFF',
            r'EBUTEIBESHEROAD': 'EBUTE/IBESHE ROAD',
            r'ALABI\s+$': 'ALABI',
        }

        for pattern, replacement in corrections.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

                                 
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    def _prepare_agent_input(self, claims: Dict, vision_flags: Dict = None) -> str:
        """Prepare clean input for the agent"""

                          
        cleaned_claims = self._clean_claims(claims)

                                     
        if cleaned_claims.get('parties', {}).get('property_owner'):
            original = cleaned_claims['parties']['property_owner']
            cleaned_claims['parties']['property_owner'] = self._clean_owner_name(original)

                                                
        user_content = (
            f"Verify this document.\n\n"
            f"CLAIMS:\n{json.dumps(cleaned_claims, indent=2)}\n\n"
        )

        if vision_flags:
                                                          
            simplified_vision = {
                "forged_pages": vision_flags.get('forged_pages', []),
                "suspicious_pages": vision_flags.get('suspicious_pages', []),
                "total_flags": vision_flags.get('total_flags', 0)
            }
            user_content += f"VISION FLAGS:\n{json.dumps(simplified_vision, indent=2)}\n\n"

        user_content += (
            "IMPORTANT: Only call tools with non-null values. "
            "If a field is null or empty, do not pass it to the tool.\n\n"
            "Start with Lagos e-GIS if you have a file_number or plot_number. "
            "If neither exists, skip to CAC registry."
        )

        return user_content

                                   

    def _execute_tool(self, name: str, args: Dict) -> Dict:
        """Execute a tool by name with given arguments, filtering nulls"""
                                                          
        filtered_args = {}
        for k, v in args.items():
            if v not in [None, "", "null", "None", [], {}]:
                filtered_args[k] = v

                                                  
        if not filtered_args and name in ["query_lagos_egis", "search_cac_registry"]:
            return {
                "error": f"No valid parameters provided for {name}",
                "available": True,
                "mock": True,
                "message": "DEMO MODE: Using mock response"
            }

        fn = TOOL_FUNCTIONS.get(name)
        if not fn:
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(**filtered_args)
        except Exception as e:
            return {"error": str(e)}

    def run(self, claims: Dict, plausibility_results: Dict = None,
            vision_flags: Dict = None) -> VerificationResult:
        """
        Run verification agent on document claims

        Args:
            claims: Extracted document fields
            plausibility_results: Results from Stage 3 checks (unused)
            vision_flags: Vision forensic flags (optional)

        Returns:
            VerificationResult with risk assessment and signals
        """
                             
        user_content = self._prepare_agent_input(claims, vision_flags)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        reasoning_trace: List[ReasoningStep] = []
        step = 0

        print("\n" + "━" * 50)
        print("  VERIFICATION AGENT (Groq / Llama 3.3 70B)")
        print("━" * 50)

        while step < self.max_steps:
            step += 1

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=2048,
                )
            except Exception as e:
                print(f"\n❌ API Error: {e}")
                                            
                return VerificationResult(
                    overall_risk="UNKNOWN",
                    signals=[],
                    reasoning_trace=reasoning_trace,
                    recommendation=f"API Error: {str(e)[:200]}. Please check your GROQ_API_KEY.",
                    squad_action="HOLD_FUNDS_IN_ESCROW",
                    trust_score=50,
                )

            msg = response.choices[0].message
            messages.append(msg)

                                           
            if not msg.tool_calls:
                print(f"\n✓ Completed in {step} steps")
                return self._parse_output(msg.content or "", reasoning_trace)

                                    
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                                                
                non_null_args = {k: v for k, v in args.items() if v not in [None, "", []]}
                print(f"\n  Step {step}: {name}({json.dumps(non_null_args)[:100]})")
                result = self._execute_tool(name, args)
                print(f"  → {json.dumps(result)[:120]}...")

                reasoning_trace.append(ReasoningStep(
                    step=step,
                    tool_called=name,
                    inputs=args,
                    output=result,
                    timestamp=datetime.now()
                ))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        print(f"\n⚠ Hit max steps ({self.max_steps})")
        return VerificationResult(
            overall_risk="UNKNOWN",
            signals=[],
            reasoning_trace=reasoning_trace,
            recommendation="Verification incomplete — max steps reached. Manual review required.",
            squad_action="HOLD_FUNDS_IN_ESCROW",
        )

    def _parse_output(self, text: str, trace: List[ReasoningStep]) -> VerificationResult:
        """Parse LLM output into VerificationResult"""
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                data = json.loads(match.group())

                               
                signals = []
                for s in data.get('signals', []):
                    signals.append(Signal(
                        source=s.get('source', 'unknown'),
                        claim_checked=s.get('claim_checked', ''),
                        result=s.get('result', ''),
                        weight=s.get('weight', 0.0),
                        severity=s.get('severity', 'LOW'),
                        explanation=s.get('explanation', '')
                    ))

                return VerificationResult(
                    overall_risk=data.get('overall_risk', 'LOW'),
                    signals=signals,
                    reasoning_trace=trace,
                    recommendation=data.get('recommendation', text[:500]),
                    squad_action=data.get('squad_action', 'HOLD_FUNDS_IN_ESCROW'),
                    trust_score=self._calculate_trust_score(signals)
                )
            except json.JSONDecodeError:
                pass

        return VerificationResult(
            overall_risk="UNKNOWN",
            signals=[],
            reasoning_trace=trace,
            recommendation=text[:1000] if text else "Agent did not return valid response",
            squad_action="HOLD_FUNDS_IN_ESCROW",
        )

    def _calculate_trust_score(self, signals: List[Signal]) -> int:
        """Calculate trust score from signals using Bayesian-like update"""
        from verification_engine.agent.config import SIGNAL_WEIGHTS

        prior = 0.75                                              
        risk_accumulator = 1.0

        for signal in signals:
            if signal.severity in ['HIGH', 'CRITICAL']:
                risk_accumulator *= (1 - signal.weight)

        trust_score = int(prior * risk_accumulator * 100)
        return max(0, min(100, trust_score))