# Import libraries needed for the project
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2  # Pre-trained model to save training time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Path to train and validation data
train_dir = r'/content/new_clas/data'
validation_dir = r'/content/new_clas/validation'

# Prepare train data with transformations for better learning
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Pixel value changed from 0-255 to 0-1
    rotation_range=40,        # Rotate image randomly
    width_shift_range=0.2,    # Move image left or right
    height_shift_range=0.2,   # Move image up or down
    shear_range=0.2,          # Shear image
    zoom_range=0.2,           # Zoom image
    horizontal_flip=True,     # Flip image left or right
    fill_mode='nearest'       # Fill missing parts after transform
)

# Validation data only rescaled, no fancy stuff
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load train and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Make every image same size
    batch_size=32,
    class_mode='categorical'  # Multiple classes here
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# MobileNetV2 used as base model. It already trained well on big dataset
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze it. No change to base model layers.

# Add our own layers on top of base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Change dimensions
    layers.Dense(512, activation='relu'),  # Layer to get features
    layers.Dropout(0.5),  # Helps model not overfit
    layers.Dense(36, activation='softmax')  # 36 classes like A-Z and 0-9
])

# Compile model. Define how it learn
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Show model details
model.summary()

# Reduce learning rate if no improvement + Stop early when no changes after some time
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,  # Number of times we go through data
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[reduce_lr, early_stopping]
)

# Save the trained model so we can use it later
model.save('hand_gesture_model_improved.h5')
print("Model saved as 'hand_gesture_model_improved.h5'.")

# Plot graphs for accuracy and loss during training
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Check how model performed on validation data
val_preds = model.predict(validation_generator)
val_preds = np.argmax(val_preds, axis=1)
val_labels = validation_generator.classes

# Confusion matrix to check correct and wrong predictions
cm = confusion_matrix(val_labels, val_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_generator.class_indices, yticklabels=train_generator.class_indices)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Show images that were misclassified
misclassified_indices = np.where(val_preds != val_labels)[0]
for i in misclassified_indices[:5]:  # First 5 misclassified images
    img = validation_generator[i][0][0]  # Get the image
    true_label = val_labels[i]
    pred_label = val_preds[i]
    plt.imshow(img)
    plt.title(f'True: {true_label}, Predicted: {pred_label}')
    plt.axis('off')
    plt.show()