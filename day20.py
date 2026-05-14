from transformers import pipeline

# 1. Load the Translation Pipeline 
# 'translation_en_to_de' na English to German (Deutsch)
print("Loading the Translation Model (Transformers)... 🤖✈️")
translator = pipeline("translation_en_to_de", model="t5-small")

def translate_this(text):
    print(f"\nOriginal (English): {text}")
    # 2. Perform Translation
    result = translator(text, max_length=40)
    translated_text = result[0]['translation_text']
    return translated_text

# 3. Test with a sentence
input_text = "Artificial Intelligence is changing the world for the better."
output = translate_this(input_text)

print(f"Translated (German): {output}")

# Bonus: Let's try another one
input_text_2 = "I am learning Machine Learning in thirty days."
output_2 = translate_this(input_text_2)
print(f"Translated (German): {output_2}")