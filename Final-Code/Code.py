import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from skimage import transform
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
import io
import os
import numpy as np
from PIL import Image, ImageEnhance
from google.colab import drive
drive.mount('/content/drive')
%cd drive/MyDrive
%cd Detection of ALL and CLL

img_size = (224, 224)
main_dir = "dataset_luk"


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.1,
    fill_mode="nearest"
)


for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    
    if os.path.isdir(subdir_path):
        count = 0 # counter to keep track of number of images processed
        print(f"Processing images in {subdir} folder:")
        
        # iterate through images in the subdirectory
        for filename in os.listdir(subdir_path):
            
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                
                if count < 2: # stop after the first two images

                    # load the image using PIL
                    img = Image.open(os.path.join(subdir_path, filename))

                    # display the original image
                    print("Original Image:")
                    display(img)

                    
                    # apply contrast adjustment to darken the leukemia stains within the cells
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(4)
                    print("Enhanced Image:")
                    display(img)

                    # resize the image
                    resized_img = img.resize(img_size)

                    # display the resized image
                    print("Resized Image:")
                    display(resized_img)

                    # normalize the pixel values
                    normalized_img = np.array(resized_img) / 255.0

                    # convert the RGB image to grayscale
                    gray_img = cv2.cvtColor(np.uint8(normalized_img*255), cv2.COLOR_RGB2GRAY)


                    # apply Otsu's thresholding to segment the image
                    _, segmented_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    # convert the segmented image back to PIL format
                    segmented_img = Image.fromarray(segmented_img)

                    # display the segmented image
                    print("Segmented Image:")
                    display(segmented_img)

                    count += 1 # increment the counter

# iterate through subdirectories in the main directory
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    
    if os.path.isdir(subdir_path):
        count = 0 # counter to keep track of number of images processed
        print(f"Processing images in {subdir} folder:")
        
        # iterate through images in the subdirectory
        for filename in os.listdir(subdir_path):
            
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                
                if count < 2: # stop after the first two images

                    # load the image using OpenCV
                    img = cv2.imread(os.path.join(subdir_path, filename))

                    # convert the image from RGB to LAB color space
                    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

                    # split the LAB image into L, A, and B channels
                    l_channel, a_channel, b_channel = cv2.split(lab_img)

                    # darken the L channel to only affect the dark purple colors
                    darkened_l_channel = cv2.addWeighted(l_channel, 0.7, np.zeros_like(l_channel), 0, 0)

                    # merge the darkened L channel with the A and B channels to get the final image
                    darkened_lab_img = cv2.merge((darkened_l_channel, a_channel, b_channel))

                    # convert the LAB image back to RGB color space
                    darkened_img = cv2.cvtColor(darkened_lab_img, cv2.COLOR_LAB2RGB)

                    # display the original image
                    print("Original Image:")
                    display(Image.fromarray(img))

                    # display the darkened image
                    print("Darkened Image:")
                    display(Image.fromarray(darkened_img))

                    # resize the image
                    resized_img = cv2.resize(darkened_img, img_size)

                    # display the resized image
                    print("Resized Image:")
                    display(Image.fromarray(resized_img))

                    # normalize the pixel values
                    normalized_img = resized_img / 255.0

                    # convert the RGB image to grayscale
                    gray_img = cv2.cvtColor(np.uint8(normalized_img*255), cv2.COLOR_RGB2GRAY)

                    # apply Otsu's thresholding to segment the image
                    _, segmented_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    # convert the segmented image back to PIL format
                    segmented_img = Image.fromarray(segmented_img)

                    # display the segmented image
                    print("Segmented Image:")
                    display(segmented_img)

                    count += 1 # increment the counter

# define the lower and upper bounds of the pink, blue, and purple colors in LAB color space
pink_lower = np.array([130, 70, 215])
pink_upper = np.array([230, 170, 255])
blue_lower = np.array([90, 120, 100])
blue_upper = np.array([180, 200, 160])
purple_lower = np.array([100, 50, 100])
purple_upper = np.array([200, 150, 200])

# iterate through subdirectories in the main directory
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    
    if os.path.isdir(subdir_path):
        count = 0 # counter to keep track of number of images processed
        print(f"Processing images in {subdir} folder:")
        
        # iterate through images in the subdirectory
        for filename in os.listdir(subdir_path):
            
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                
                if count < 2: # stop after the first two images

                    # load the image using OpenCV
                    img = cv2.imread(os.path.join(subdir_path, filename))

                    # convert the image from RGB to LAB color space
                    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

                    # define masks for the pink, blue, and purple colors
                    pink_mask = cv2.inRange(lab_img, pink_lower, pink_upper)
                    blue_mask = cv2.inRange(lab_img, blue_lower, blue_upper)
                    purple_mask = cv2.inRange(lab_img, purple_lower, purple_upper)

                    # combine the masks into a single mask
                    mask = cv2.bitwise_or(pink_mask, blue_mask)
                    mask = cv2.bitwise_or(mask, purple_mask)

                    # apply the mask to the original image
                    masked_img = cv2.bitwise_and(img, img, mask=mask)

                    # display the original image
                    print("Original Image:")
                    display(Image.fromarray(img))

                    # display the masked image
                    print("Masked Image:")
                    display(Image.fromarray(masked_img))

                    # resize the image
                    resized_img = cv2.resize(masked_img, img_size)

                    # display the resized image
                    print("Resized Image:")
                    display(Image.fromarray(resized_img))

                    # normalize the pixel values
                    normalized_img = resized_img / 255.0

                    # convert the RGB image to grayscale
                    gray_img = cv2.cvtColor(np.uint8(normalized_img*255), cv2.COLOR_RGB2GRAY)

                    # apply Otsu's thresholding to segment the image
                    _, segmented_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    # convert the segmented image back to PIL format
                    segmented_img = Image.fromarray(segmented_img)

                    # display the segmented image
                    print("Segmented Image:")
                    display(segmented_img)

                    count += 1 # increment the counter


# iterate through subdirectories in the main directory
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    
    if os.path.isdir(subdir_path):
        count = 0 # counter to keep track of number of images processed
        print(f"Processing images in {subdir} folder:")
        
        # iterate through images in the subdirectory
        for filename in os.listdir(subdir_path):
            
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                
                if count < 2: # stop after the first two images

                    # load the image using OpenCV
                    img = cv2.imread(os.path.join(subdir_path, filename))

                    # convert the image from RGB to LAB color space
                    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

                    # calculate the mean of the L, A, and B channels
                    l_mean, a_mean, b_mean = cv2.mean(lab_img)[:3]

                    # display the unique color
                    print(f"Unique color in image {filename}: L={l_mean:.2f}, A={a_mean:.2f}, B={b_mean:.2f}")

                    count += 1 # increment the counter


# iterate through subdirectories in the main directory
for subdir in os.listdir(main_dir):
    subdir_path = os.path.join(main_dir, subdir)
    
    if os.path.isdir(subdir_path):
        count = 0 # counter to keep track of number of images processed
        print(f"Processing images in {subdir} folder...")
        
        # iterate through images in the subdirectory
        for filename in os.listdir(subdir_path):
            
            if filename.endswith(".jpg") or filename.endswith(".jpeg"):
                
                    # load the image using PIL
                    img = Image.open(os.path.join(subdir_path, filename))
                    
                    # apply contrast adjustment to darken the leukemia stains within the cells
                    contrast = ImageEnhance.Contrast(img)
                    img = contrast.enhance(4)

                    # resize the image
                    resized_img = img.resize(img_size)

                    # normalize the pixel values
                    normalized_img = np.array(resized_img) / 255.0

                    # convert the RGB image to grayscale
                    gray_img = cv2.cvtColor(np.uint8(normalized_img*255), cv2.COLOR_RGB2GRAY)


                    # apply Otsu's thresholding to segment the image
                    _, segmented_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                    # convert the segmented image back to PIL format
                    segmented_img = Image.fromarray(segmented_img)
    print(f"Processed images in {subdir} folder.")

from tensorflow.keras import datasets, layers, models
import tensorflow as tf

import os
class_names = os.listdir("dataset_luk")

train_ds = tf.keras.utils.image_dataset_from_directory(
  "dataset_luk",
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(224, 224),
  batch_size=16)
testing_ds = tf.keras.utils.image_dataset_from_directory(
  "dataset_luk",
  validation_split=.2,
  subset="validation",
  seed=123,
  image_size=(224, 224),
  batch_size=16)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

num_classes = len(class_names)
img_height, img_width = 224,224
model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
epochs=10
history = model.fit(
  train_ds,
  validation_data=testing_ds,
  epochs=epochs
)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


from skimage.io import imread
img_array = []
img_loc = os.listdir("dataset_luk")
for label in img_loc:
    images = os.listdir("dataset_luk/" + label) 
    for img in images:
        img_array.append(imread("dataset_luk/" + label+"/"+img))
        # print("dataset_luk/" + label+"/"+img)
predictions = model.predict(np.array(img_array))
score = tf.nn.softmax(predictions[0])

predictions = model.predict(np.array([img_array[-5]]))
predictions

