from crewai import Agent, Task, Crew, Process
import os

print("--- Step 1: Hiring the AI Agents ---")

# Agent 1: The Tech Researcher
researcher = Agent(
    role='Senior Tech Researcher',
    goal='Analyze the latest trends in Machine Learning and AI',
    backstory='You are an expert AI researcher working at a top tech company. You find accurate and up-to-date information.',
    verbose=True,
    allow_delegation=False
)

# Agent 2: The Tech Blogger
writer = Agent(
    role='Tech Blogger',
    goal='Write an engaging blog post based on the researcher finding',
    backstory='You are a famous tech blogger known for simplifying complex AI topics for beginners. Your writing is highly engaging.',
    verbose=True,
    allow_delegation=False
)

print("✅ AI Team (Researcher & Writer) hired successfully!")

print("\n--- Step 2: Assigning the Tasks ---")

task1 = Task(
    description='Research the top 3 AI advancements in the current year and summarize them.',
    expected_output='A bulleted list of the top 3 AI advancements with brief, factual explanations.',
    agent=researcher
)

task2 = Task(
    description='Using the summary from the researcher, write a 2-paragraph blog post.',
    expected_output='A 2-paragraph highly engaging blog post explaining the advancements without technical jargon.',
    agent=writer
)

print("✅ Tasks assigned to respective agents!")

print("\n--- Step 3: Forming the Crew and Starting Work ---")

# The Crew manages the agents and tasks
tech_crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential # Task 1 must finish before Task 2 starts
)

print("🚀 Crew is set up and ready to kickoff!")
print("⚠️ (To execute this fully in a real environment, set your OPENAI_API_KEY locally)")

# Code to run the crew (Commented out to prevent API errors without a key)
# result = tech_crew.kickoff()
# print("\n--- FINAL OUTPUT DELIVERED BY THE CREW ---")
# print(result)