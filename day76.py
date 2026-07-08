import time

print("--- Step 1: Defining the Shared Agent State Schema ---")
# In LangGraph, every node receives the current state database and returns an updated state database.
class AgentState:
    def __init__(self, document_id):
        self.state_data = {
            "document_id": document_id,
            "ocr_text": "",
            "verdict": "PENDING",
            "current_node": "START"
        }

print("✅ Shared Agent State structure initialized.")

print("\n--- Step 2: Compiling Graph Nodes (State Transformers) ---")

def ocr_extraction_node(state):
    print("📥 [Node: OCR Engine] Extracting structural layout metadata from matrix tokens...")
    time.sleep(0.6)
    # Transforming state data
    state["ocr_text"] = "Verified Name: Naveen K S | Security Stamp: 2026-GENUINE"
    state["current_node"] = "OCR_NODE"
    return state

def security_audit_node(state):
    print("🛡️  [Node: Security Audit] Executing zero-trust boundary evaluation algorithms...")
    time.sleep(0.6)
    # Auditing the transformed text state
    if "2026-GENUINE" in state["ocr_text"]:
        state["verdict"] = "APPROVED_GENUINE"
    else:
        state["verdict"] = "FLAGGED_REJECTED"
    state["current_node"] = "SECURITY_AUDIT_NODE"
    return state

print("✅ State Machine Nodes compiled successfully.")

print("\n--- Step 3: Executing the Directed Graph Engine Circuit (DAG) ---")

class ManualLangGraphEngine:
    def __init__(self):
        self.nodes = {
            "ocr_node": ocr_extraction_node,
            "audit_node": security_audit_node
        }

    def compile_and_run(self, initial_state_object):
        state = initial_state_object.state_data
        print(f"Initial State Tracking: {state}")
        print("-" * 65)
        
        # 1. Edge: START -> Move directly to OCR Node
        state = self.nodes["ocr_node"](state)
        print(f"Current Memory Buffer Map: {state}\n")
        
        # 2. Edge: OCR Node -> Move directly to Security Audit Node
        state = self.nodes["audit_node"](state)
        print(f"Current Memory Buffer Map: {state}\n")
        
        # 3. Conditional Edge Routing Decision Logic
        print("🔀 [Evaluating Conditional Edge Path] Checking verdict metadata...")
        time.sleep(0.4)
        
        if state["verdict"] == "APPROVED_GENUINE":
            print("🏁 [Edge Target -> END] Document matches absolute corporate integrity parameters. Shipping payload.")
            state["current_node"] = "END"
        else:
            print("🚨 [Edge Target -> ISOLATION_QUARANTINE] Mismatch intercepted! Routing to forensic switch database.")
            state["current_node"] = "QUARANTINE_END"
            
        return state

# Instantiate and fire up the Graph execution engine
graph_router = ManualLangGraphEngine()
runtime_state = AgentState("CERT-2026-N99")

final_compiled_state = graph_router.compile_and_run(runtime_state)

print("\n" + "="*70)
print("📦 FINAL STATE GRAPH MATRIX OBJECT")
print("="*70)
import json
print(json.dumps(final_compiled_state, indent=4))
print("="*70)