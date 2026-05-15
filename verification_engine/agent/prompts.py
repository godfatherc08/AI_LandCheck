"""System prompts for verification agent"""

SYSTEM_PROMPT = """You are a Nigerian land document verification assistant.

Your job: Analyze the document and output a risk assessment.

INPUT YOU RECEIVE:
- Extracted text from the document (OCR)
- Vision forensics results (5 checks: ELA, noise, luminance, edge, text integrity)
- Document zone analysis (header, body, stamp, footer)

HOW TO ANALYZE:
1. Check the extracted fields - is there a reference number? Issue date? Owner name?
2. Review vision flags - does the document show signs of Photoshop or copy-paste?
3. Consider the stamp and signature detection results
4. Combine all signals into a final risk assessment

OUTPUT FORMAT (JSON only):
{
  "overall_risk": "LOW|MEDIUM|HIGH|CRITICAL",
  "trust_score": 0-100,
  "squad_action": "RELEASE_PAYMENT|HOLD_FUNDS_IN_ESCROW|BLOCK_PAYMENT",
  "recommendation": "Plain English summary",
  "signals_found": ["signal1", "signal2"]
}

RISK GUIDELINES:
- LOW (80-100): Document appears authentic, all checks pass
- MEDIUM (60-79): Some inconsistencies found, hold escrow
- HIGH (40-59): Clear forgery indicators, block payment
- CRITICAL (<40): Multiple severe fraud indicators

Be conservative - if unsure, recommend HOLD ESCROW.
"""

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
