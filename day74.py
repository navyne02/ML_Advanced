import json
import os

print("--- Step 1: Architecting the Stateful Agent Memory Manager ---")

class StatefulAgentMemory:
    def __init__(self, storage_file="agent_long_term_vault.json"):
        self.storage_file = storage_file
        self.short_term_buffer = []  # In-memory stack for current conversation
        self.long_term_profile = {}  # Facts persisted across historical runs
        self._load_long_term_vault()

    def _load_long_term_vault(self):
        """Loads historical facts from local disk if existing"""
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                self.long_term_profile = json.load(f)
            print("💾 [Long-Term Vault] Historical user facts successfully hydrated into memory.")
        else:
            self.long_term_profile = {}
            print("🆕 [Long-Term Vault] No previous facts found. Initializing a clean canvas.")

    def remember_short_term(self, role, message):
        """Appends the immediate interaction to the current thread buffer"""
        self.short_term_buffer.append({"role": role, "content": message})
        print(f"🧠 [Short-Term Buffer] Appended interaction from [{role}]")

    def extract_and_save_fact(self, key, value):
        """Autonomously extracts a critical fact and commits it to persistent storage"""
        self.long_term_profile[key] = value
        with open(self.storage_file, 'w') as f:
            json.dump(self.long_term_profile, f, indent=4)
        print(f"🔒 [Long-Term Vault] Fact committed to disk -> Key: {key} | Value: {value}")

    def get_agent_context(self):
        """Combines short-term thread and long-term knowledge profiles for the LLM"""
        return {
            "historical_profile": self.long_term_profile,
            "current_thread": self.short_term_buffer
        }

print("✅ Agent Memory Engine compiled.")

print("\n--- Step 2: Simulating Conversation Run 1 (Fact Learning & Storage) ---")

# First run: Agent meets user and learns personal context
agent_brain = StatefulAgentMemory()

# Conversational exchange
agent_brain.remember_short_term("User", "Hi, my name is Naveen and I am currently optimizing an Asus A15 setup.")
agent_brain.remember_short_term("Agent", "Hello Naveen! I have noted your technical development configuration.")

# Agent identifies a long-term critical preference asset and persists it
agent_brain.extract_and_save_fact("user_name", "Naveen K S")
agent_brain.extract_and_save_fact("hardware_spec", "Asus TUF Gaming A15")

print("\n--- Step 3: Simulating System Reset ---")
# Simulating the program closing or the server restarting (Wiping short-term buffer)
del agent_brain
print("⚠️ Short-term RAM cleared. Server restarted.")

print("\n--- Step 4: Simulating Conversation Run 2 (Memory Recall) ---")
# Second run: Agent wakes up in a completely new session, but reloads the long-term vault
new_agent_session = StatefulAgentMemory()

print(f"\nFinal Compiled Agent Context for next prompt generation:")
complete_context = new_agent_session.get_agent_context()
print(json.dumps(complete_context, indent=4))