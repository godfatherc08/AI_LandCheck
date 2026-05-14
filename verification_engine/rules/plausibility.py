"""
Stage 3: Plausibility Checks - Pure Python Rules Engine
Deterministic validation of extracted document fields
"""

import re
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class CheckSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class PlausibilityResult:
    check_name: str
    result: str
    weight: float
    explanation: str
    severity: CheckSeverity
    expected_format: Optional[str] = None
    actual_value: Optional[str] = None
    suggestion: Optional[str] = None


class StateNormalizer:
    """Normalize Nigerian state names to standard format"""

    STATE_MAPPING = {
        "LAGOS STATE": "LAGOS",
        "LAGOS": "LAGOS",
        "LAGOS ST": "LAGOS",
        "LAG": "LAGOS",
        "ABUJA FCT": "ABUJA",
        "FCT": "ABUJA",
        "ABUJA": "ABUJA",
        "OGUN STATE": "OGUN",
        "OGUN": "OGUN",
        "RIVERS STATE": "RIVERS",
        "RIVERS": "RIVERS",
        "KADUNA STATE": "KADUNA",
        "KADUNA": "KADUNA",
        "ANAMBRA STATE": "ANAMBRA",
        "ANAMBRA": "ANAMBRA",
    }

    @classmethod
    def normalize(cls, state: str) -> Optional[str]:
        if not state:
            return None
        state_upper = state.upper().strip()
        for key, value in cls.STATE_MAPPING.items():
            if key in state_upper or state_upper in key:
                return value
        return state_upper


class FieldCleaner:
    """Clean and validate extracted field values"""

    @staticmethod
    def clean_local_government(value: str) -> Dict:
        if not value:
            return {"value": None, "cleaned": False}

        cleaned = re.sub(r'\s+', ' ', value.upper())
        cleaned = re.sub(r'LOCAL\s+GOVERNMENT\s+AREA', '', cleaned)
        cleaned = re.sub(r'L\.?G\.?A\.?', '', cleaned)
        cleaned = re.sub(r'LAGOS\s+STATE', '', cleaned)
        cleaned = re.sub(r'SCALE.*$', '', cleaned)
        cleaned = cleaned.strip()

        valid_lgas = [
            "IKEJA", "ALIMOSHO", "APAPA", "ETI-OSA", "IKORODU",
            "LAGOS ISLAND", "LAGOS MAINLAND", "SURULERE", "MUSHIN",
            "OSHODI", "BADAGRY", "EPE", "IBEJU-LEKKI"
        ]

        for valid in valid_lgas:
            if valid in cleaned or cleaned in valid:
                return {"value": valid, "cleaned": True, "original": value}

        return {"value": cleaned if len(cleaned) > 3 else None, "cleaned": len(cleaned) > 3, "original": value}

    @staticmethod
    def clean_scale(value: str) -> Dict:
        if not value:
            return {"value": None, "cleaned": False}

        match = re.search(r'(\d+)\s*:\s*(\d+)', value)
        if match:
            return {"value": f"{match.group(1)}:{match.group(2)}", "cleaned": True, "original": value}

        return {"value": None, "cleaned": False, "original": value, "error": "Invalid format"}


class PlausibilityChecker:
    """Deterministic plausibility checks for Nigerian land documents"""

    def __init__(self):
        self.state_patterns = {
            "LAGOS": {"cof": r'^LGS/\d{4}/(COO|RA)/\d{5}$', "example": "LGS/2018/COO/04821"},
            "ABUJA": {"cof": r'^FCT/\d{4}/RA/\d{5}$', "example": "FCT/2019/RA/00123"},
            "OGUN": {"cof": r'^OGN/\d{4}/COO/\d{5}$', "example": "OGN/2020/COO/00567"},
        }

        self.term_standards = {"RESIDENTIAL": 99, "COMMERCIAL": 99, "AGRICULTURAL": 25}

    def check_file_number_format(self, file_number: str, state: str) -> PlausibilityResult:
        if not file_number:
            return PlausibilityResult("file_number_format", "N/A", 0.0, "No file number provided", CheckSeverity.LOW)

        normalized_state = StateNormalizer.normalize(state) if state else None

        if normalized_state and normalized_state in self.state_patterns:
            pattern = self.state_patterns[normalized_state]["cof"]
            if re.match(pattern, file_number.upper()):
                return PlausibilityResult("file_number_format", "PASS", 0.0, f"Valid format for {normalized_state}", CheckSeverity.LOW)
            else:
                return PlausibilityResult("file_number_format", "FAIL", 0.7, f"Invalid format for {normalized_state}", CheckSeverity.MEDIUM,
                                         self.state_patterns[normalized_state]["example"], file_number,
                                         f"Should match {self.state_patterns[normalized_state]['example']}")

        return PlausibilityResult("file_number_format", "WARN", 0.3, "Cannot validate - state unknown", CheckSeverity.LOW, actual_value=file_number)

    def check_issue_date_validity(self, issue_date: Any) -> PlausibilityResult:
        if not issue_date:
            return PlausibilityResult("issue_date_validity", "N/A", 0.0, "No issue date", CheckSeverity.LOW)

        parsed_date = None
        if isinstance(issue_date, dict) and issue_date.get('formatted'):
            try:
                parsed_date = datetime.strptime(issue_date['formatted'], "%Y-%m-%d").date()
            except:
                pass
        elif isinstance(issue_date, str):
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
                try:
                    parsed_date = datetime.strptime(issue_date, fmt).date()
                    break
                except:
                    continue

        if not parsed_date:
            return PlausibilityResult("issue_date_validity", "WARN", 0.3, f"Could not parse date: {issue_date}", CheckSeverity.LOW, actual_value=str(issue_date))

        today = date.today()
        if parsed_date > today:
            return PlausibilityResult("issue_date_validity", "FAIL", 0.85, f"Date {parsed_date} is in the future", CheckSeverity.HIGH, actual_value=str(parsed_date))

        years_old = today.year - parsed_date.year
        if years_old > 25:
            return PlausibilityResult("issue_date_validity", "WARN", 0.4, f"Document is {years_old} years old", CheckSeverity.MEDIUM, actual_value=str(parsed_date))

        if parsed_date.weekday() >= 5:
            return PlausibilityResult("issue_date_validity", "FAIL", 0.55, f"Date falls on weekend", CheckSeverity.MEDIUM, actual_value=str(parsed_date))

        return PlausibilityResult("issue_date_validity", "PASS", 0.0, f"Date {parsed_date} is valid", CheckSeverity.LOW, actual_value=str(parsed_date))

    def check_lga_validity(self, lga: str, state: str) -> PlausibilityResult:
        if not lga:
            return PlausibilityResult("lga_validity", "N/A", 0.0, "No LGA provided", CheckSeverity.LOW)

        cleaned = FieldCleaner.clean_local_government(lga)
        normalized_state = StateNormalizer.normalize(state) if state else None

        if not normalized_state:
            return PlausibilityResult("lga_validity", "WARN", 0.3, f"State unknown, cannot validate LGA '{lga}'", CheckSeverity.LOW, actual_value=lga)

        if cleaned.get('value'):
            return PlausibilityResult("lga_validity", "PASS", 0.0, f"LGA '{cleaned['value']}' is valid", CheckSeverity.LOW, actual_value=lga)

        return PlausibilityResult("lga_validity", "FAIL", 0.55, f"LGA '{lga}' not recognized", CheckSeverity.MEDIUM, actual_value=lga)

    def check_term_length(self, term_years: Any, land_use: Optional[str] = None) -> PlausibilityResult:
        if not term_years:
            return PlausibilityResult("term_length", "N/A", 0.0, "No term length (may not be C of O)", CheckSeverity.LOW)

        try:
            term_int = int(term_years)
        except:
            return PlausibilityResult("term_length", "WARN", 0.35, f"Could not parse '{term_years}'", CheckSeverity.LOW, actual_value=str(term_years))

        land_use_upper = land_use.upper() if land_use else "RESIDENTIAL"
        standard = self.term_standards.get(land_use_upper, 99)

        if term_int == standard:
            return PlausibilityResult("term_length", "PASS", 0.0, f"Term {term_int} years matches standard", CheckSeverity.LOW, str(standard), str(term_int))
        elif 90 <= term_int <= 105:
            return PlausibilityResult("term_length", "WARN", 0.25, f"Term {term_int} years near standard {standard}", CheckSeverity.LOW, str(standard), str(term_int))
        else:
            return PlausibilityResult("term_length", "FAIL", 0.6, f"Term {term_int} years deviates from standard {standard}", CheckSeverity.MEDIUM, str(standard), str(term_int))

    def run_all_checks(self, extracted_data: Dict) -> Dict:
        fields = extracted_data.get('extracted_fields', extracted_data)

        results = [
            self.check_file_number_format(fields.get('file_number') or fields.get('ref_number'), fields.get('state')),
            self.check_issue_date_validity(fields.get('issue_date')),
            self.check_lga_validity(fields.get('local_government') or fields.get('local_government_area'), fields.get('state')),
            self.check_term_length(fields.get('term_years'), fields.get('land_use_zoning'))
        ]

        total_weight = sum(r.weight for r in results if r.result == "FAIL")
        failed = [r for r in results if r.result == "FAIL"]
        warnings = [r for r in results if r.result == "WARN"]

        if total_weight >= 1.5:
            overall = "VERY_UNLIKELY"
            recommendation = "Multiple logical inconsistencies - likely fraudulent"
        elif total_weight >= 0.8:
            overall = "SUSPICIOUS"
            recommendation = "Notable inconsistencies - verify with original source"
        elif total_weight >= 0.3:
            overall = "PARTIALLY_QUESTIONABLE"
            recommendation = "Minor inconsistencies - may be clerical errors"
        else:
            overall = "PLAUSIBLE"
            recommendation = "Document passes plausibility checks"

        return {
            "plausibility_checks": [self._result_to_dict(r) for r in results],
            "summary": {
                "total_checks": len(results),
                "passed": len([r for r in results if r.result == "PASS"]),
                "failed": len(failed),
                "warnings": len(warnings),
                "total_suspicion_weight": round(total_weight, 2),
                "overall_plausibility": overall,
                "recommendation": recommendation
            }
        }

    def _result_to_dict(self, result: PlausibilityResult) -> Dict:
        return {
            "check": result.check_name,
            "result": result.result,
            "weight": result.weight,
            "severity": result.severity.value,
            "explanation": result.explanation,
            "expected_format": result.expected_format,
            "actual_value": result.actual_value,
            "suggestion": result.suggestion
        }


def run_plausibility_checks(extracted_data: Dict) -> Dict:
    """Convenience function to run all plausibility checks"""
    checker = PlausibilityChecker()
    return checker.run_all_checks(extracted_data)