import os
import glob
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# Data Paths
INPUT_DATA = '/kaggle/input/face-mask-detection/images'
ANNOTATIONS = "/kaggle/input/face-mask-detection/annotations"
IMAGES = os.listdir(INPUT_DATA)
OUTPUT_DATA =  '.'

# Parsing XML annotations
def parse_object(element):
    coordinates = {item.tag: int(item.text) for item in element.iter() if item.tag in ['xmin', 'ymin', 'xmax', 'ymax']}
    return {**coordinates, 'name': element.find('name').text}

def parse_annotation(file):
    tree = ET.parse(file)
    root = tree.getroot()
    data = {item.tag: item.text for item in root.iter() if item.tag in ['filename', 'width', 'height', 'depth']}
    objects = [parse_object(obj) for obj in root.iter('object')]
    return [{**data, **obj} for obj in objects]

# Load dataset
dataset = [item for sublist in [parse_annotation(file) for file in glob.glob(f"{ANNOTATIONS}/*.xml")] for item in sublist]
df = pd.DataFrame(dataset)

# Split dataset
TEST_IMAGE = 'maksssksksss0'
df_test = df[df["filename"] == TEST_IMAGE]
df = df[df["filename"] != TEST_IMAGE]

# Create folders for train, test and validation data
LABELS = df['name'].unique()
for label in LABELS:
    for d in ['train', 'test', 'val']:
        os.makedirs(os.path.join(OUTPUT_DATA, d, label), exist_ok=True)

# Crop images
def crop_image(image_path, xmin, ymin, xmax, ymax):
    img = Image.open(image_path)
    return img.crop((xmin, ymin, xmax, ymax))

def extract_faces(image_info):
    df_img = image_info[image_info['filename'] == image[:-4]][['xmin', 'ymin', 'xmax', 'ymax', 'name']]
    return [(crop_image(os.path.join(INPUT_DATA, image), *df_img.iloc[i]), df_img['name'].iloc[i]) for i in range(len(df_img))]

# Get cropped faces
faces = [face for sublist in [extract_faces(df[df['filename'] == image[:-4]]) for image in IMAGES] for face in sublist]

# Split data
data_splits = {label: train_test_split([face for face in faces if face[1] == label], test_size=0.2, random_state=42) for label in LABELS}
for label in LABELS:
    data_splits[label][1], val = train_test_split(data_splits[label][1], test_size=0.5, random_state=42)
    data_splits[label].append(val)

# Save images
def save_image(image, image_name, path, dataset_type, label):
    image.save(os.path.join(path, dataset_type, label, f'{image_name}.png'))

for dataset_type in ['train', 'test', 'val']:
    for label in LABELS:
        for image, image_name in data_splits[label][['train', 'test', 'val'].index(dataset_type)]:
            save_image(image, image_name, OUTPUT_DATA, dataset_type, label)

# Define model
model = Sequential()
model.add(Conv2D(16, kernel_size = 3,  padding='same', activation = 'relu', input_shape = (35,35,3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(32, kernel_size = 3,  padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Conv2D(64, kernel_size = 3,  padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(3, activation = 'softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy', 'Recall', 'Precision', 'AUC'])

# Image data generators
datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.1, shear_range=0.2, width_shift_range=0.1, height_shift_range=0.1, rotation_range=4, vertical_flip=False)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = datagen.flow_from_directory(directory='/kaggle/working/train', target_size = (35,35), class_mode="categorical", batch_size=8, shuffle=True)
val_generator = val_datagen.flow_from_directory(directory='/kaggle/working/val', target_size = (35,35), class_mode="categorical", batch_size=8, shuffle=True)
test_generator = val_datagen.flow_from_directory(directory='/kaggle/working/test', target_size = (35,35), class_mode="categorical", batch_size=8, shuffle=False)

# Callbacks contd.
lrr = ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# Train model
history = model.fit(train_generator, validation_data=val_generator, steps_per_epoch=100, validation_steps=50, epochs=50, callbacks=[early_stopping, lrr])

# Plot the model training history
plt.figure(figsize=(15, 7))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'])

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'])

plt.show()

# Evaluate model on the test data
test_loss, test_accuracy, test_recall, test_precision, test_auc = model.evaluate(test_generator)

# Print metrics
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
print(f"Test recall: {test_recall}")
print(f"Test precision: {test_precision}")
print(f"Test AUC: {test_auc}")

# Predict labels
preds = model.predict(test_generator)
y_pred = np.argmax(preds, axis=-1)

# Print confusion matrix
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))

# Print classification report
print('Classification Report')
target_names = list(test_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

# Save model
model.save("mask_detection_model.h5")
