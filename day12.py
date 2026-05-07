import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. Load Data (CIFAR-10 color images)
print("Loading CIFAR-10 dataset... 🚛✈️")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Build the CNN Model
model = models.Sequential([
    # Layer 1: Convolution + Pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 2: More Convolution
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Layer 3: Dense Layers (Brain part)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10) # 10 classes
])

# 3. Compile and Train
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("\nTraining CNN (Ithu konjam extra time edukkum because of color images)... 🧠")
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))

# 4. Evaluate Accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nAccuracy on Test Images: {test_acc*100:.2f}%')

# 5. Visual Check
plt.figure(figsize=(5,5))
plt.imshow(test_images[0])
prediction = model.predict(test_images[0:1])
predicted_class = class_names[prediction.argmax()]
plt.title(f"Actual: {class_names[test_labels[0][0]]} | AI Predicts: {predicted_class}")
plt.show()