import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

print("AI is collecting news data... 📰")

# 1. Create a dummy dataset (Real world-la neenga pd.read_csv('news.csv') use pannuvinga)
data = {
    'text': [
        "Government announces new tax reforms for the middle class.", 
        "Shocking! Drink this magical water to lose 10kg in 1 day!!!", 
        "ISRO successfully launches the new communication satellite.",
        "Aliens found living secretly under the Eiffel Tower!",
        "The stock market saw a 2% increase today after tech earnings.",
        "Send this to 10 people or your WhatsApp will be deleted tomorrow!",
        "New AI model breaks record in medical image analysis.",
        "Eat garlic and honey to become completely immortal."
    ],
    'label': ['REAL', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE', 'REAL', 'FAKE']
}

df = pd.DataFrame(data)

# 2. Split data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)

# 3. Initialize TF-IDF Vectorizer (Text-ah numbers-ah mathurom)
# stop_words='english' -> The, is, in maathiri unnecessary words-ah remove pannidum
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# 4. Initialize and Train the AI Model
print("AI is analyzing the difference between facts and lies... 🕵️‍♂️")
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# 5. Predict and Evaluate
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"\n--- AI Evaluation ---")
print(f"Model Accuracy: {round(score*100, 2)}%\n")

# 6. Real-time Testing Custom Engine
print("--- Let's Test Custom Headlines! ---")
custom_news = [
    "NASA confirms the discovery of a new exoplanet with water.",
    "Click this link to win a free iPhone 15 Pro Max immediately!!!"
]

custom_tfidf = tfidf_vectorizer.transform(custom_news)
predictions = pac.predict(custom_tfidf)

for text, prediction in zip(custom_news, predictions):
    if prediction == 'FAKE':
        print(f"🛑 FAKE NEWS DETECTED: '{text}'")
    else:
        print(f"✅ REAL NEWS DETECTED: '{text}'")