import numpy as np

print("--- Step 1: Simulating RAG Production Logs (Query, Context, Output) ---")

# Real-world-la evaluation panna intha 4 elements thevai
rag_logs = [
    {
        "query": "What is the grievance handling system setup at Rank Projects?",
        "ground_truth": "Rank Projects follows a 3-step escalation pathway ending at the Project Director.",
        "retrieved_context": "Construction labor grievance handling system at Rank Projects follows a 3-step escalation pathway ending at the Project Director.",
        "ai_generated_answer": "It follows a 3-step escalation pathway ending at the Project Director."
    },
    {
        "query": "What is the policy for e-waste disposal in Salem?",
        "ground_truth": "Mandates authorized electronic waste processing before the end of each quarter.",
        "retrieved_context": "Environmental impact assessments in Salem mandate authorized electronic waste processing before the end of each quarter.",
        "ai_generated_answer": "E-waste must be recycled every month at any local shop." # Hallucination / Wrong Answer!
    }
]

# 2. Advanced Evaluation Logic Core Simulation
def evaluate_rag_performance(logs):
    print("\n--- Step 2: Running Automated Metric Evaluation Rules ---")
    
    for idx, log in enumerate(logs):
        print(f"\nEvaluating Sample {idx + 1}:")
        print(f"User Query : '{log['query']}'")
        
        # Simulating Jaccard Similarity / Token Overlap for Faithfulness
        # How much of the AI answer is derived from the retrieved context?
        context_words = set(log['retrieved_context'].lower().split())
        answer_words = set(log['ai_generated_answer'].lower().split())
        
        overlap = context_words.intersection(answer_words)
        faithfulness_score = len(overlap) / max(len(answer_words), 1)
        
        # Simulating Ground Truth Accuracy (Answer Relevance)
        truth_words = set(log['ground_truth'].lower().split())
        truth_overlap = truth_words.intersection(answer_words)
        relevance_score = len(truth_overlap) / max(len(truth_words), 1)
        
        print(f"🎯 Metric - Faithfulness (No Hallucination) : {faithfulness_score * 100:.1f}%")
        print(f"🎯 Metric - Answer Relevance (Accuracy)      : {relevance_score * 100:.1f}%")
        
        if faithfulness_score < 0.4 or relevance_score < 0.4:
            print("🛑 ALERT: Potential Hallucination or Low Quality Output Detected!")
        else:
            print("✅ PASS: Output aligns perfectly with enterprise knowledge base.")

evaluate_rag_performance(rag_logs)