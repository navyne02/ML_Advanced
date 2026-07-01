import json
from pydantic import BaseModel, Field, ValidationError

print("--- Step 1: Defining the Strict Output Schema (Pydantic Model) ---")

# Defining exactly what data structure we expect from the AI for a Verification Audit
class DocumentAuditResult(BaseModel):
    is_authentic: bool = Field(description="True if the certificate/document is genuine, False otherwise.")
    confidence_score: float = Field(description="A floating point percentage score between 0.0 and 1.0.")
    extracted_name: str = Field(description="The exact name listed on the document.")
    detected_flags: list[str] = Field(description="List of anomalies spotted. Empty list if clean.")

print("✅ Strict Data Law Enforced for AI Outputs.")

print("\n--- Step 2: Simulating Brittle/Broken LLM Response (The Incident) ---")

# Simulating a broken JSON output from a raw LLM (Missing a key, data types swapped)
# 1. 'confidence_score' is sent as text instead of float! 
# 2. 'detected_flags' key is missing completely!
broken_llm_payload = """
{
    "is_authentic": false,
    "confidence_score": "High confidence around 85 percent", 
    "extracted_name": "Naveen Kumar"
}
"""
print("Incoming Raw LLM Output to be audited:")
print(broken_llm_payload.strip())

print("\n--- Step 3: Running the Guardrail Validation Engine ---")

try:
    # Attempting to parse the raw string into our strict schema structure
    parsed_json = json.loads(broken_llm_payload)
    validated_data = DocumentAuditResult(**parsed_json)
    print("🟢 [SUCCESS] Output matches schema perfectly.")
    
except ValidationError as e:
    print("🚨 [GUARDRAIL BLOCK] AI Output failed structural verification guidelines!")
    print("-" * 65)
    print("Captured Validation Errors to feed back into LLM Self-Correction Loop:")
    
    # Extracting exact error logs to heal the pipeline
    errors = e.errors()
    error_feedback_log = []
    for err in errors:
        loc = err['loc'][0]
        msg = err['msg']
        print(f"👉 Field Field: [{loc}] | Issue: {msg}")
        error_feedback_log.append(f"Field '{loc}' failed validation due to: {msg}")
        
    print("-" * 65)
    
    print("\n--- Step 4: Activating Self-Correction Loop Flow ---")
    print("Sending error diagnostics back to LLM to force compliance... 🔄")
    
    # Simulating the corrected LLM output after reading the guardrail error logs
    corrected_llm_payload = """
    {
        "is_authentic": false,
        "confidence_score": 0.85,
        "extracted_name": "Naveen Kumar",
        "detected_flags": ["Signature Font Mismatch", "Digital Stamp Altered"]
    }
    """
    
    # Re-validating the corrected payload
    final_json = json.loads(corrected_llm_payload)
    final_validated_output = DocumentAuditResult(**final_json)
    
    print("\n🎉 [RE-VALIDATION SUCCESSFUL] AI successfully healed its output schema!")
    print(f"Final Production Ready Object JSON: {final_validated_output.model_dump_json(indent=2)}")