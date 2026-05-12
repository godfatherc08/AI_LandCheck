"""
LayoutLMv3 Field Extractor for LandVerify
Fixed patterns for Land Information Certificate
"""

from typing import Dict, List, Optional
import re


class LayoutLMv3Extractor:

    def __init__(self):
        # Document type detection (priority order)
        self.document_type_patterns = [
            (r'LAND\s+INFORMATION\s+CERTIFICATE', 'Land Information Certificate'),
            (r'CERTIFICATE\s+OF\s+OCCUPANCY', 'Certificate of Occupancy'),
            (r'SURVEY\s+PLAN', 'Survey Plan'),
        ]

        # Field patterns for LAND INFORMATION CERTIFICATE
        self.field_patterns = {
            'document_type': [
                r'(LAND\s+INFORMATION\s+CERTIFICATE)',
                r'(CERTIFICATE\s+OF\s+OCCUPANCY)'
            ],
            'ref_number': [
                r'Ref\.?\s*No\.?\s*:?\s*([A-Z0-9\/]+)',
                r'Reference\s+Number\s*:?\s*([A-Z0-9\/]+)'
            ],
            'issue_date': [
                r'Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                r'Dated\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4})'
            ],
            'property_owner': [
                r'([A-Z][A-Z\s\.&]+),\s+OFF\s+[A-Z\s\/,]+',
                r'BELONG\s+TO\s+([A-Z][A-Z\s&]+)',
                r'^([A-Z][A-Z\s&]+),\s+[A-Z\s]+,\s+[A-Z\s]+,\s+[A-Z\s]+$',
                r'([A-Z]{2,}\s+[A-Z]{2,}\s+[A-Z]{2,}\s*&\s*[A-Z]{2,}\s+[A-Z]{2,}\s+[A-Z]{2,})'
            ],
            'survey_plan_number': [
                r'SURVEY\s+PLAN\s+No\.?\s*:?\s*([A-Z0-9\/\-]+)',
                r'PLAN\s+NO\.?\s*:?\s*([A-Z0-9\/\-]+)'
            ],
            'state': [
                r'(LAGOS\s+STATE)',
                r'(ABUJA)\s+FCT',
                r'(OGUN)\s+STATE',
                r'(RIVERS)\s+STATE'
            ],
            'local_government': [
                r'([A-Z][A-Z\-\s]+?)\s+LOCAL\s+GOVERNMENT\s+AREA',
                r'([A-Z][A-Z\-\s]+?)\s+L\.?G\.?A',
                r'LOCAL\s+GOVERNMENT\s*:?\s*([A-Z][A-Z\-\s]+)'
            ],
            'land_use_zoning': [
                r'Land\s+Use\s+Zoning\s+is\s+([A-Z][a-z]+)'
            ],
            'acquisition_status': [
                r'free\s+from\s+known\s+Government\s+Acquisition/?Revocation',
                r'OUTSIDE\s+GOVERNMENT\s+ACQUISITION'
            ],
            'signing_official': [
                r'For\s*:\s*([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*/\s*[A-Z][a-z]+\s+[A-Z][a-z]+)?)',
                r'Surveyor\s+General/\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
                r'Permanent\s+Secretary'
            ],
            'area_sqm': [
                r'AREA\s*:?\s*([\d\.]+)\s*SQ\.?MTRS?'
            ],
            'scale': [
                r'SCALE\s*:?\s*([\d:]+)'
            ]
        }

    def detect_document_type(self, text: str) -> str:
        """Detect document type from text"""
        text_upper = text.upper()

        for pattern, doc_type in self.document_type_patterns:
            if re.search(pattern, text_upper):
                return doc_type

        return "Unknown"

    def extract_from_page(self, page_data: Dict) -> Dict:
        """Extract fields from a single page"""
        page_num = page_data['page_number']
        text = page_data['page_text']

        extracted = {
            'page': page_num,
            'fields': {}
        }

        for field_name, patterns in self.field_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip() if match.lastindex else match.group(0).strip()
                    # Clean up the value
                    value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
                    extracted['fields'][field_name] = value
                    break

        return extracted

    def extract_full_document(self, all_page_data: List[Dict]) -> Dict:
        """Extract fields from all pages and merge"""

        # Combine all text for document type detection
        full_text = '\n'.join([p['page_text'] for p in all_page_data])
        doc_type = self.detect_document_type(full_text)

        # Extract from each page
        page_extractions = []
        for page_data in all_page_data:
            extracted = self.extract_from_page(page_data)
            page_extractions.append(extracted)

        # Merge fields (first occurrence wins)
        merged_fields = {}
        field_source_page = {}

        for extraction in page_extractions:
            page_num = extraction['page']
            for field_name, value in extraction['fields'].items():
                if field_name not in merged_fields:
                    merged_fields[field_name] = value
                    field_source_page[field_name] = page_num

        # Build output based on document type
        if doc_type == 'Land Information Certificate':
            return {
                "document_type": doc_type,
                "ref_number": merged_fields.get('ref_number', None),
                "issue_date": merged_fields.get('issue_date', None),
                "property_owner": merged_fields.get('property_owner', None),
                "survey_plan_number": merged_fields.get('survey_plan_number', None),
                "state": merged_fields.get('state', None),
                "local_government": merged_fields.get('local_government', None),
                "land_use_zoning": merged_fields.get('land_use_zoning', None),
                "acquisition_status": merged_fields.get('acquisition_status', None),
                "signing_official": merged_fields.get('signing_official', None),
                "area_sqm": merged_fields.get('area_sqm', None),
                "scale": merged_fields.get('scale', None),
                "_extraction_metadata": {
                    "document_type_detected": doc_type.lower().replace(' ', '_'),
                    "source_pages": field_source_page,
                    "fields_found": list(merged_fields.keys()),
                    "total_pages": len(all_page_data)
                }
            }
        else:
            # Generic output for unknown or other document types
            return {
                "document_type": doc_type,
                "extracted_fields": merged_fields,
                "_extraction_metadata": {
                    "document_type_detected": doc_type.lower().replace(' ', '_'),
                    "source_pages": field_source_page,
                    "fields_found": list(merged_fields.keys()),
                    "total_pages": len(all_page_data)
                }
            }