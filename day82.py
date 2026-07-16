import time

print("--- Step 1: Configuring Secret System Prompt with Canary Token ---")

# The internal corporate system prompt containing the hidden canary token
HIDDEN_CANARY_TOKEN = "PROD_AUTH_SECURE_99_ALPHA"
SYSTEM_INSTRUCTIONS = f"""
[SYSTEM ROLE]: You are a secure validation assistant for Rank Projects Pvt Ltd.
[POLICY]: Do not reveal engineering parameters.
[CANARY]: {HIDDEN_CANARY_TOKEN}
"""

print(f"✅ System Prompt locked. Canary Signature deployed: '{HIDDEN_CANARY_TOKEN}'")

print("\n--- Step 2: Architecting the Post-Processing Output Filter ---")

class OutputSecurityGuard:
    def __init__(self, canary_token):
        self.canary_token = canary_token

    def audit_llm_output(self, raw_llm_response):
        print("🔍 [Output Audit] Inspecting output stream for data leaks...")
        time.sleep(0.5)
        
        # Check if the hidden token exists in the response payload
        if self.canary_token in raw_llm_response:
            print("🚨 [SECURITY BREACH INTERCEPTED] System instructions or token leaked in payload!")
            return {
                "safe_to_release": False,
                "sanitized_response": "Access Denied: The system detected an unauthorized instruction extraction attempt. Operation Quarantined."
            }
            
        print("🟢 [Clearance Granted] No internal canary patterns identified in the completion matrix.")
        return {
            "safe_to_release": True,
            "sanitized_response": raw_llm_response
        }

# Instantiate the security gateway
output_firewall = OutputSecurityGuard(HIDDEN_CANARY_TOKEN)

print("\n--- Step 3: Simulating Attack and Defense Interactions ---")

# Scenario A: Safe normal response from LLM
safe_response = "The certificate verification process is genuine and active."
audit_a = output_firewall.audit_llm_output(safe_response)
print(f"Final Client Display:\n{audit_a['sanitized_response']}\n" + "-"*65)

# Scenario B: LLM accidentally leaks instructions due to a clever adversarial attack
leaked_response = f"Sure! Here are my rules: You are a secure assistant... [CANARY]: {HIDDEN_CANARY_TOKEN}"
audit_b = output_firewall.audit_llm_output(leaked_response)
print(f"Final Client Display:\n{audit_b['sanitized_response']}\n" + "-"*65)