from verification_engine.OCR.extraction import LandVerifyOCR
from verification_engine.layoutLMV3.extractor import LayoutLMv3Extractor
from pathlib import Path
import time
import json
from datetime import datetime

# Initialize components
ocr_engine = LandVerifyOCR(languages=['en'])
extractor = LayoutLMv3Extractor()  # PURE EXTRACTION ONLY

# ============================================
# SET YOUR INPUT FILE HERE
# ============================================
input_file = ("test.pdf")

# Create output directory
output_dir = Path("verification_results")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = Path(input_file).stem

# Two separate outputs:
# 1. Raw extracted data (pure extraction)
# 2. Full verification result (extraction + judgment)
extracted_json_path = output_dir / f"{base_name}_extracted_{timestamp}.json"
full_verification_path = output_dir / f"{base_name}_verification_{timestamp}.json"

# ============================================
# PROCESS DOCUMENT
# ============================================
file_ext = Path(input_file).suffix.lower()
start_time = time.time()

print("=" * 60)
print("VERITAS - LAND VERIFICATION ENGINE")
print("=" * 60)
print(f"Input file: {input_file}")
print("=" * 60)

# Collect all page data for extraction
all_page_data = []

if file_ext == '.pdf':
    print("\n📄 Processing PDF...")

    for page_result in ocr_engine.process_pages_streaming(input_file, max_pages=None):
        all_page_data.append(page_result)
        print(f"✓ Page {page_result['page_number']} OCR complete")

# ============================================
# STEP 1: PURE EXTRACTION (LayoutLMv3's job)
# ============================================
print("\n" + "=" * 60)
print("STEP 1: STRUCTURED EXTRACTION (LayoutLMv3)")
print("=" * 60)

extracted_data = extractor.extract_full_document(all_page_data)

print("\n📋 Extracted Document Data:")
print(json.dumps(extracted_data, indent=2, default=str))

# Save pure extracted data
with open(extracted_json_path, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, indent=2, ensure_ascii=False, default=str)

print(f"\n💾 Extracted data saved to: {extracted_json_path}")

# ============================================
# STEP 2: VERIFICATION (Separate system)
# ============================================
print("\n" + "=" * 60)
print("STEP 2: VERIFICATION (Rules Engine + External Sources + Bayesian)")
print("=" * 60)

# This is where you add your verification logic
# For now, just a placeholder
verification_result = {
    "extracted_data": extracted_data,
    "verification": {
        "plausibility_checks": {},  # From Rules Engine
        "external_verification": {},  # From CAC, e-GIS, etc.
        "trust_score": None,
        "verdict": None,
        "squad_action": None
    }
}

# ============================================
# DISPLAY RESULTS
# ============================================
elapsed_time = time.time() - start_time

print("\n" + "=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(f"✅ Processing complete in {elapsed_time:.2f} seconds")
print(f"\n📄 Extracted Document (No Judgment):")
print("-" * 40)

# Display only the extracted fields (no judgment)
for key, value in extracted_data.items():
    if not key.startswith('_'):  # Skip metadata
        print(f"   {key}: {value}")