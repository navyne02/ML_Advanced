import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# 1. Sample Text Data (Neenga vera ethavathu periya text kooda kudukkalam)
text = "machine learning is amazing. ai is the future of technology. learning ai is fun."
text = text.lower()

# 2. Character Mapping
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# 3. Create Sequences
maxlen = 10 # 10 characters-ah paathu 11th character-ah predict panna porom
step = 1
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

# Vectorization (Numbers-ah mathurom)
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 4. Build the Model
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# 5. Training and Generating Text
print("AI is learning to write... ✍️")
model.fit(x, y, batch_size=128, epochs=50, verbose=0)

def generate_text(seed, length=20):
    generated = seed
    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed):
            x_pred[0, t, char_indices[char]] = 1.
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]
        
        generated += next_char
        seed = seed[1:] + next_char
    return generated

# Test the AI
start_seed = "machine le" # Must be 10 chars
print(f"\nSeed: {start_seed}")
print(f"AI Writes: {generate_text(start_seed)}")