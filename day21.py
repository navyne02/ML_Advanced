import spacy

# 1. Load the pre-trained English model (Small version)
print("Loading spaCy NLP model... 🧠")
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Model illana download panna sollum
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    # 2. Process the text
    doc = nlp(text)
    
    print(f"\n--- Original Text ---\n{text}\n")
    print("--- Detected Entities ---")
    
    # 3. Iterate through entities
    for ent in doc.ents:
        print(f"Entity: {ent.text: <15} | Label: {ent.label_: <10} | Description: {spacy.explain(ent.label_)}")

# 4. Test with a real-world scenario
sample_text = """
Naveen started his 30-day AI challenge in Salem, Tamil Nadu. 
He is learning technologies like TensorFlow and spaCy. 
Google and Microsoft are hiring AI engineers in May 2026. 
He plans to finish the project by next Friday.
"""

extract_entities(sample_text)