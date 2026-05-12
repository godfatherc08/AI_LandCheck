# verification_engine/OCR/ocr_run.py

from verification_engine.OCR.extraction import LandVerifyOCR
from verification_engine.layoutLMV3.extractor import LayoutLMv3Extractor
from pathlib import Path
import time
import json
from datetime import datetime

# Initialize with vision enabled
print("=" * 60)
print("VERITAS - LAND VERIFICATION ENGINE")
print("=" * 60)
print("Vision forgery detection: ENABLED (sensitivity: high)")
print("=" * 60)

ocr_engine = LandVerifyOCR(
    languages=['en'],
    enable_vision=True,
    sensitivity='high'
)

extractor = LayoutLMv3Extractor()

# Input file
input_file = "test.pdf"
file_ext = Path(input_file).suffix.lower()

# Create output directory
output_dir = Path("verification_results")
output_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = Path(input_file).stem

start_time = time.time()

# Process based on file type
if file_ext == '.pdf':
    print(f"\n📄 Processing PDF: {input_file}")

    # Get full analysis with vision
    result = ocr_engine.get_full_analysis(input_file, max_pages=None)

    # Extract fields using LayoutLMv3
    extracted_data = extractor.extract_full_document(result['page_data'])

    # Display vision summary
    print("\n" + "=" * 60)
    print("VISION ANALYSIS RESULTS")
    print("=" * 60)

    if result['vision_summary']:
        vs = result['vision_summary']
        print(f"Pages analyzed: {vs['pages_analyzed']}")
        print(f"Likely Forged pages: {vs['forged_pages']}")
        print(f"Suspicious pages: {vs['suspicious_pages']}")

        if vs['forged_pages']:
            print("\n⚠️ FORGERY INDICATORS DETECTED:")
            for flag in vs['all_flags']:
                print(f"   • {flag['check'].upper()}: {flag['reason'][:100]}...")

    # Display extracted data
    print("\n" + "=" * 60)
    print("EXTRACTED DOCUMENT DATA")
    print("=" * 60)

    for key, value in extracted_data.items():
        if not key.startswith('_') and value:
            print(f"   {key}: {value}")

    # Build final output
    final_output = {
        "verification_id": f"VRT-{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "source_file": input_file,
        "processing_time_seconds": round(time.time() - start_time, 2),
        "vision_analysis": result['vision_summary'],
        "extracted_data": extracted_data,
        "total_pages": len(result['page_data']),
        "total_tokens": len(result['tokens'])
    }

    # Save to JSON
    output_path = output_dir / f"{base_name}_full_analysis_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 Full analysis saved to: {output_path}")
    print(f"✅ Completed in {final_output['processing_time_seconds']:.2f} seconds")

else:
    print(f"Unsupported file type: {file_ext}")