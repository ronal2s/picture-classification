# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython.display import Image
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


# %%
filename = "images/chicken/1625259245623.jpeg"

# %% [markdown]
# # Load the deep learning model
#

# %%
# Deep learning model weights -pre-trained

mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()


# %%
# Creating a model
# Training a model
## Test or validate
# Predict

# %% [markdown]
# # Pre-processing of the image

# %%
filename = "images/chicken//1625259245667.jpeg"
Image(filename=filename, width=224, height=224)

img = image.load_img(filename, target_size=(224, 224))
resized_img = image.img_to_array(img)
final_image = np.expand_dims(resized_img, axis=0)  # Need fourth dimension ¿?
final_image = tf.keras.applications.mobilenet_v2.preprocess_input(final_image)
final_image.shape

# %% [markdown]
# # Make prediction

# %%
predictions = mobile.predict(final_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)
# print(results[0][0])


# %%
plt.imshow(img)


# %%
