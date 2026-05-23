import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

print("--- Step 1: Initializing Agent Core Processing Units ---")
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def call_local_llm(prompt_text):
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. Agent A: The Specialized Researcher
def researcher_agent(topic_query):
    print(f"\n[Agent 1: Researcher] 🕵️‍♂️ Gathering core data points for: '{topic_query}'")
    research_prompt = f"""
    Act as an AI Researcher. Provide 3 factual technical words or bullet points related to the topic.
    Topic: {topic_query}
    Facts:
    """
    raw_research = call_local_llm(research_prompt)
    return raw_research

# 4. Agent B: The Executive Reviewer & Writer
def executive_writer_agent(collected_research, topic_query):
    print(f"\n[Agent 2: Writer] ✍️ Processing research insights and generating final summary...")
    writer_prompt = f"""
    Act as an Executive Writer. Turn these raw technical facts into a one-sentence formal summary about {topic_query}.
    Raw Facts: {collected_research}
    Formal Summary:
    """
    final_report = call_local_llm(writer_prompt)
    return final_report

print("\n--- Step 2: Triggering Multi-Agent Orchestration Pipeline ---")
# Defining a topic relevant to your active interest vectors
target_topic = "Machine Learning Decision Trees and Data Analysis"

# Execution Flow (Agent 1 passes outputs directly to Agent 2)
start_time = time.time()

research_output = researcher_agent(target_topic)
print(f"-> Agent 1 Raw Output: {research_output}")

final_summary_report = executive_writer_agent(research_output, target_topic)

print("\n--- Final Multi-Agent Combined Report Assembly ---")
print(f"🚀 Execution Complete in {time.time() - start_time:.2f} seconds!")
print(f"📋 Final System Delivery:\n\"{final_summary_report}\"")