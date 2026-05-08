import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# 1. Load Pre-trained Model (MobileNetV2)
# include_top=False na, kadaisi classifier layer-ah thookidu nu artham
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                               include_top=False,
                                               weights='imagenet')

# 2. Freeze the Base (Pazhaiya moolaiyila irukkura knowledge-ah matha vandaam)
base_model.trainable = False

# 3. Add our own Custom Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid') # Binary Classification (Cat vs Dog or 0 vs 1)
])

# 4. Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

print("\nAI moolai ready! Ippo namma periya models-ah namma chinna task-ku use panna thayar. 🚀")

# 5. Let's test with a random image (as we don't have a dataset loaded today)
random_img = np.random.rand(1, 160, 160, 3)
prediction = model.predict(random_img)
print(f"Prediction for random image: {prediction[0][0]}")