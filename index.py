import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Define the data directories
data_dir = '../input/brain-mri-images-for-brain-tumor-detection'
img_size = (128, 128)

# Create data generators for loading and preprocessing images
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# train_generator = train_datagen.flow_from_directory(
#     data_dir,
#     target_size=img_size,
#     batch_size=16,
#     class_mode='binary',  # Modify class_mode to 'binary'
#     subset='training',
#     shuffle=True,
#     color_mode='grayscale'  # Specify color mode for grayscale images
# )

# # Calculate the number of validation samples
# val_samples = len(train_generator.filenames[train_generator.indexes[:len(train_generator.filenames)//5]])

# # Adjust the batch size for the validation set
# val_batch_size = int(np.ceil(val_samples / len(val_generator)))

# Calculate the number of validation samples
val_samples = len(train_generator) // 5

# Adjust the batch size for the validation set
val_batch_size = int(np.ceil(val_samples / len(val_generator)))

# # Create the validation data generator with the adjusted batch size
# val_generator = train_datagen.flow_from_directory(
#     data_dir,
#     target_size=img_size,
#     batch_size=val_batch_size,
#     class_mode='binary',
#     subset='validation',
#     shuffle=False,
#     color_mode='grayscale'
# )

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=16,
    class_mode='input',  # Modify class_mode to 'binary'
    subset='training',
    shuffle=True,
    color_mode='grayscale'  # Specify color mode for grayscale images
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=val_batch_size,
    class_mode='binary',  # Modify class_mode to 'binary'
    subset='validation',
    shuffle=False,
    color_mode='grayscale'  # Specify color mode for grayscale images
)


print(len(train_generator))
print(len(val_generator))

# Define the U-Net++ model
def unet_plus(input_shape=(128, 128, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up5 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv4))
    merge5 = Concatenate(axis=3)([conv3, up5])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up6 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = Concatenate(axis=3)([conv2, up6])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv1, up7])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = Conv2D(1, 1, activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = unet_plus(input_shape=(128, 128, 1))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the callbacks
checkpoint_path = 'unet_plus.keras'
checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[checkpoint_cb]
)

# Load the best model weights after training
model.load_weights(checkpoint_path)

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_generator)
print(f'Validation loss: {val_loss:.4f}')
print(f'Validation accuracy: {val_accuracy:.4f}')

# Reshape the target labels to match the output shape of the model
y_true = np.expand_dims(val_generator.labels, axis=-1)

# Ensure that the number of samples in y_true and y_pred_classes are the same
y_true = y_true[:val_samples]

# Predict on the validation data
y_pred = model.predict(val_generator)
y_pred = y_pred[:val_samples]

# Convert predictions to binary classes
y_pred_classes = (y_pred > 0.5).astype(int)

# Calculate confusion matrix
confusion_mtx = confusion_matrix(y_true.ravel(), y_pred_classes.ravel())
class_names = ['No tumor', 'Tumor']

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_true.ravel(), y_pred_classes.ravel(), target_names=class_names))
