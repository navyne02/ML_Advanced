import time

print("--- Step 1: Building the Behavior Tree Engine Framework ---")

class Node:
    """Base class for all nodes in the Behavior Tree"""
    def tick(self, agent):
        raise NotImplementedError

class Selector(Node):
    """OR Logic: Runs children until one SUCCEEDS"""
    def __init__(self, children):
        self.children = children

    def tick(self, agent):
        for child in self.children:
            if child.tick(agent) == "SUCCESS":
                return "SUCCESS"
        return "FAILURE"

class Sequence(Node):
    """AND Logic: Runs children until one FAILS"""
    def __init__(self, children):
        self.children = children

    def tick(self, agent):
        for child in self.children:
            if child.tick(agent) == "FAILURE":
                return "FAILURE"
        return "SUCCESS"

print("✅ Behavior Tree Framework Core Ready.")

print("\n--- Step 2: Creating Conditions and Actions (Leaf Nodes) ---")

# Condition Nodes
class IsHealthLow(Node):
    def tick(self, agent):
        if agent["health"] < 40:
            print("  🧐 [Condition] Health is critically low!")
            return "SUCCESS"
        return "FAILURE"

class IsEnemyVisible(Node):
    def tick(self, agent):
        if agent["enemy_visible"]:
            print("  🧐 [Condition] Enemy spotted in visual range!")
            return "SUCCESS"
        return "FAILURE"

# Action Nodes
class UseHealPotion(Node):
    def tick(self, agent):
        print("  ⚡ [Action] Drinking Elixir... Health Restored to 100%!")
        agent["health"] = 100
        return "SUCCESS"

class AttackEnemy(Node):
    def tick(self, agent):
        print("  ⚔️ [Action] Charging towards the target! Attack executed.")
        return "SUCCESS"

class PatrolBase(Node):
    def tick(self, agent):
        print("  🛡️ [Action] No threats found. Calmly patrolling the layout boundaries.")
        return "SUCCESS"

print("✅ Leaf Nodes (Conditions/Actions) compiled.")

print("\n--- Step 3: Architecting the Brain Strategy Matrix ---")

# Constructing the Tree Layout
# Strategy: IF health is low -> HEAL. ELSE IF enemy is visible -> ATTACK. ELSE -> PATROL.
ai_brain = Selector([
    Sequence([IsHealthLow(), UseHealPotion()]),
    Sequence([IsEnemyVisible(), AttackEnemy()]),
    PatrolBase()
])

print("🤖 AI Strategy Tree Initialized.")

print("\n--- Step 4: Simulating Dynamic Game State Ticks ---")

# Our Warrior Agent's starting attributes
warrior = {
    "health": 30,          # Low health initially
    "enemy_visible": True
}

print(f"Game State 1: {warrior}")
print("Ticking AI Brain...")
ai_brain.tick(warrior)

print("\n" + "-"*40)
# State changes after healing
warrior["enemy_visible"] = True
print(f"Game State 2: {warrior}")
print("Ticking AI Brain...")
ai_brain.tick(warrior)

print("\n" + "-"*40)
# Enemy eliminated
warrior["enemy_visible"] = False
print(f"Game State 3: {warrior}")
print("Ticking AI Brain...")
ai_brain.tick(warrior)