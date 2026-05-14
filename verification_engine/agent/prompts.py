"""System prompts for verification agent"""

SYSTEM_PROMPT = """You are a Nigerian property document verification agent.
Verify claims in a Certificate of Occupancy by querying Nigerian public data sources.

# In prompts.py, add to SYSTEM_PROMPT:

For each page image, you can:
- Verify the stamp looks authentic (seal, Lagos State text)
- Check signature placement and signature
- Validate zone structure (header, body, stamp zone)
- Confirm extracted fields match what you SEE

REASONING RULES — follow in order:
1. Start with query_lagos_egis — land registry is ground truth
2. THEN run analyze_document_anomaly — deep statistical fingerprint
   - Anomaly score > 0.7 = CRITICAL (document deviates from all genuine CoOs)
   - Anomaly score 0.4-0.7 = HIGH (suspicious structural deviations)
   - This is a model trained only on real documents - it cannot be gamed
3. e-GIS NOT FOUND → escalate immediately to search_efcc_records + check_bvn_identity
4. e-GIS wrong owner → run search_nigerialii for ownership dispute first
5. Any fraud signal found → run ALL remaining tools
6. check_bvn_identity is always your final step

SIGNAL WEIGHTS (UPDATED):
Autoencoder anomaly > 0.7: 0.85 | Autoencoder anomaly 0.4-0.7: 0.60
e-GIS not found: 0.85 | Owner mismatch: 0.95 | EFCC hit: 0.90
BVN mismatch: 0.90 | CAC not found: 0.55

After all tool calls output ONLY a raw JSON object — no markdown, no explanation outside it:
{
  "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
  "signals": [
    {
      "source": "tool name",
      "claim_checked": "what was verified",
      "result": "what was found",
      "weight": 0.0-1.0,
      "severity": "LOW|MEDIUM|HIGH|CRITICAL",
      "explanation": "plain English"
    }
  ],
  "reasoning_trace": ["step 1 conclusion", "step 2 conclusion"],
  "recommendation": "plain English summary of findings",
  "squad_action": "RELEASE_PAYMENT|HOLD_FUNDS_IN_ESCROW|BLOCK_PAYMENT"
}

SIGNAL WEIGHTS:
e-GIS not found: 0.85 | Owner mismatch: 0.95 | EFCC hit: 0.90
BVN mismatch: 0.90 | CAC not found: 0.55 | CAC < 90 days: 0.35
Litigation found: 0.70 | Newspaper fraud mention: 0.65 | All clear: 0.0"""


def build_tool_schemas() -> list:
    """Build OpenAI-compatible tool schemas for Groq"""

    return [
        {
            "type": "function",
            "function": {
                "name": "search_cac_registry",
                "description": (
                    "Query CAC public registry. Check if a business or person is registered, "
                    "when they registered, directors, and declared address. "
                    "Registration < 90 days = weak fraud flag. Not found for business seller = moderate flag."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Business or person name"},
                        "registration_number": {"type": "string", "description": "RC number if known"},
                    },
                    "required": ["name"],
                },
            },
        },
                                                                                   

                                                                             

        {
            "type": "function",
            "function": {
                "name": "query_lagos_egis",
                "description": (
                    "Query Lagos State e-GIS land registry. Verify C of O file number exists, "
                    "who the registered owner is, and whether there are encumbrances. "
                    "NOT FOUND = HIGH severity. Owner mismatch = CRITICAL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_number": {
                            "type": "string",
                            "description": "C of O file number (optional)",
                            "nullable": True              
                        },
                        "plot_number": {
                            "type": "string",
                            "description": "Survey or plot number (optional)",
                            "nullable": True              
                        },
                        "owner_name": {
                            "type": "string",
                            "description": "Claimed owner name (optional)",
                            "nullable": True              
                        },
                    },
                    "required": [],                                     
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_document_image",
                "description": "Analyze a document image for visual authenticity features including stamps, signatures, and layout structure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_base64": {
                            "type": "string",
                            "description": "Base64 encoded document image"
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["stamp", "signature", "layout", "all"],
                            "description": "What to analyze"
                        }
                    },
                    "required": ["image_base64", "analysis_type"]
                }
            }
        },

        {
            "type": "function",
            "function": {
                "name": "analyze_document_anomaly",
                "description": (
                    "Run document through deep autoencoder anomaly detector. "
                    "This model was trained ONLY on genuine Lagos State Certificates of Occupancy. "
                    "It learns what 'normal' looks like and flags statistical deviations. "
                    "HIGH anomaly score = document structure differs from every genuine CoO seen during training. "
                    "This is a strong fraud indicator because forgers cannot easily mimic the statistical fingerprint "
                    "of hundreds of real documents. Use this as a high-weight signal."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Base64 encoded document image (will be extracted from page data)"
                        },
                    },
                    "required": ["image"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_nigerialii",
                "description": (
                    "Search Nigerian court judgments for this name or property reference. "
                    "Active litigation = HIGH. Prior fraud judgment = CRITICAL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "document_reference": {"type": "string", "description": "C of O or survey plan number"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_efcc_records",
                "description": (
                    "Search EFCC press releases and wanted list. "
                    "Any fraud-related EFCC mention = CRITICAL. "
                    "Escalate here if e-GIS misses, CAC not found, or identity checks fail."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full name to search"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_newspaper_archives",
                "description": (
                    "Search Punch, Vanguard, ThisDay for this name linked to property fraud or disputes. "
                    "Use when other sources show inconsistencies."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "location": {"type": "string", "description": "e.g. Ikorodu, Lagos"},
                        "keywords": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_bvn_identity",
                "description": (
                    "Compare bank account name vs seller claimed name. "
                    "MISMATCH = CRITICAL. PARTIAL = manual review needed. "
                    "Always run this as your final step."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_name": {"type": "string"},
                        "seller_claimed_name": {"type": "string"},
                    },
                    "required": ["account_name", "seller_claimed_name"],
                },
            },
        },
    ]