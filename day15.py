import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load IMDB Dataset (Top 5000 words)
print("Loading Movie Reviews... 🍿")
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# 2. Sequence Padding (Ellaa review-aiyum orey length-ku mathurom)
max_review_length = 500
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

# 3. Build LSTM Model
model = Sequential([
    Embedding(top_words, 32, input_length=max_review_length),
    SpatialDropout1D(0.2),
    LSTM(100), # The Memory Layer
    Dense(1, activation='sigmoid') # 1 = Positive, 0 = Negative
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Train the Model
print("\nAI is reading reviews and learning emotions... 🧠")
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

# 5. Evaluate
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAccuracy: {scores[1]*100:.2f}%")