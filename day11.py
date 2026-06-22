import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load Data (MNIST dataset - 0 to 9 numbers)
print("Loading handwritten digits data... 🔢")
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize data (Numbers-ah 0 to 1 kulla mathurom - faster training)
X_train, X_test = X_train / 255.0, X_test / 255.0

# 2. Build the Neural Network Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),    # 28x28 image-ah single line-ah mathurom
    layers.Dense(128, activation='relu'),    # Hidden Layer with 128 neurons
    layers.Dropout(0.2),                     # Overfitting thadukka chinna break
    layers.Dense(10, activation='softmax')   # Output Layer (10 numbers: 0-9)
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the AI (Moolai padikkuthu!)
print("\nAI is training on the numbers... 🧠")
model.fit(X_train, y_train, epochs=5)

# 5. Test the AI
print("\nTesting the AI on new images...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nFinal Accuracy: {test_acc*100:.2f}%')

# Visual Check: AI prediction vs Actual
prediction = model.predict(X_test)
plt.imshow(X_test[0], cmap='gray')
plt.title(f"Actual: {y_test[0]} | AI Predicts: {prediction[0].argmax()}")
plt.show() #vday100