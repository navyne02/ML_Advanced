import numpy as np
import time

print("--- Step 1: Enforcing Semantic Route Profiles (Agent Vectors) ---")

# Defining isolated Agent Destinations
def execute_grievance_agent(query):
    return f"💼 [Grievance Agent Active] Processing labor/construction compliance ticket for query: '{query}'"

def execute_ocr_agent(query):
    return f"🔍 [OCR Audit Agent Active] Launching image entropy and stamp inspection checks for query: '{query}'"

def execute_default_agent(query):
    return f"🤖 [General Bot Active] Handling standard corporate query: '{query}'"

# Simulating semantic vector space coordinates (3 Dimensions for calculation ease)
# Dimension 1: HR/Labor context, Dimension 2: Image/OCR/Security context, Dimension 3: General FAQ
route_matrix = {
    "grievance_route": {"vector": np.array([0.95, 0.05, 0.10]), "executor": execute_grievance_agent},
    "ocr_route":       {"vector": np.array([0.05, 0.90, 0.15]), "executor": execute_ocr_agent}
}

print("✅ Semantic map profiles and destination nodes registered successfully.")

print("\n--- Step 2: Architecting the Semantic Router Core Logic ---")

class SemanticAgentRouter:
    def __init__(self, routes):
        self.routes = routes

    def calculate_cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def route_query(self, user_query, query_vector):
        print(f"📥 Intercepting Query Vector: '{user_query}'")
        time.sleep(0.4)
        
        best_route = None
        highest_score = -1.0
        
        # Calculate mathematical closeness across all registered agent nodes
        for route_name, route_data in self.routes.items():
            score = self.calculate_cosine_similarity(query_vector, route_data["vector"])
            print(f"   📊 Target Closeness to [{route_name}]: {score:.4f}")
            
            if score > highest_score:
                highest_score = score
                best_route = route_name
                
        # Hard threshold enforcement (If match score is lower than 0.70, route to default)
        if highest_score >= 0.70:
            print(f"🔀 [Match Confirmed] Rerouting switch fabric to -> {best_route}")
            return self.routes[best_route]["executor"](user_query)
        else:
            print("🔀 [Low Similarity Matrix] Routing to General Corporate Support Node.")
            return execute_default_agent(user_query)

# Instantiate our custom gateway switch
network_gate = SemanticAgentRouter(route_matrix)

print("\n--- Step 3: Simulating Live Operational Queries ---")

# Scenario A: User asks about workers or grievances
query_a = "What is the procedure to log a construction worker complaint?"
vector_a = np.array([0.88, 0.12, 0.08]) # High dimension 1 (Labor)
report_a = network_gate.route_query(query_a, vector_a)
print(f"📡 Result: {report_a}\n" + "-"*65)

# Scenario B: User asks about fake files or OCR scanning
query_b = "Scan this image file to see if the certificate digital stamp is altered."
vector_b = np.array([0.02, 0.95, 0.10]) # High dimension 2 (OCR/Security)
report_b = network_gate.route_query(query_b, vector_b)
print(f"📡 Result: {report_b}\n" + "-"*65)