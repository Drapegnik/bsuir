#!/usr/bin/env python
# coding: utf-8

# # ml lab5

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 0. load dataset

# In[6]:


get_ipython().system("chmod +x ./download_data.sh ")
get_ipython().system("./download_data.sh")


# ### 1. train / test split

# In[7]:


import os
import glob


# In[8]:


image_shape = (128, 128, 3)

DATASET_PATH = "./data/dogs-vs-cats/train"


def get_label(name):
    label = "dog" if "dog" in name else "cat"
    return [name, label]


data_df = pd.DataFrame(
    data=[
        get_label(os.path.basename(image_path))
        for image_path in glob.glob(f"{DATASET_PATH}/*.jpg")
    ],
    columns=["name", "label"],
)
data_df.head()


# In[23]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(data_df, test_size=0.3, random_state=42)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


def generate_data(datagen, df, batch_size=64):
    return datagen.flow_from_dataframe(
        df,
        DATASET_PATH,
        x_col="name",
        y_col="label",
        target_size=image_shape[:-1],
        class_mode="binary",
        batch_size=batch_size,
    )


datagen = ImageDataGenerator(rescale=1.0 / 255)
train_generator = generate_data(datagen, train_df)
test_generator = generate_data(datagen, test_df)


# In[10]:


from random import randrange

fig, axs = plt.subplots(1, 10, figsize=(20, 2))
for ax in axs:
    i = randrange(train_df.shape[0])
    ax.imshow(plt.imread(f'{DATASET_PATH}/{train_df["name"][i]}'))
    ax.set_title(train_df["label"][i])
    ax.axis("off")


# ### 2. build network

# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Activation,
    BatchNormalization,
    Dropout,
    Flatten,
)


def build_model():
    model = Sequential()

    model.add(Conv2D(32, 3, padding="same", input_shape=image_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, 3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    return model


# In[7]:


model = build_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[20]:


print(f'Train accuracy: {history.history["accuracy"][-1]*100:.2f}%')


# In[28]:


def plot(_history):
    plt.figure(figsize=(8, 6))
    plt.plot(_history.history["accuracy"], "r")
    plt.plot(_history.history["val_accuracy"], "b")
    plt.legend(["Training", "Validation"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")


# In[38]:


plot(history)


# ### 2. data augmentation

# In[40]:


datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    width_shift_range=0.1,
    height_shift_range=0.1,
)
train_generator = generate_data(datagen, train_df)

datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = generate_data(datagen, test_df)


# In[41]:


model = build_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[46]:


history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[49]:


print(f'Train accuracy: {history.history["accuracy"][-1]*100:.2f}%')


# In[48]:


plot(history)


# > augmentation prevents overfitting

# ### 4. Try VGG16

# In[21]:


from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

image_shape = (224, 224, 3)

pre_trained_model = VGG16(
    input_shape=image_shape, include_top=False, weights="imagenet"
)


for i, layer in enumerate(pre_trained_model.layers):
    layer.trainable = i > 20

last_layer = pre_trained_model.get_layer("block5_pool")
last_output = last_layer.output

x = GlobalAveragePooling2D()(last_output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(pre_trained_model.input, x)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()


# In[24]:


datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    width_shift_range=0.1,
    height_shift_range=0.1,
)
train_generator = generate_data(datagen, train_df)

datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = generate_data(datagen, test_df)


# In[25]:


history = model.fit(
    train_generator,
    epochs=1,
    validation_data=test_generator,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[34]:


print(f'Train accuracy: {history.history["accuracy"][-1]*100:.2f}%')


# In[35]:


results = model.evaluate(test_generator)
print(f"Test accuracy: {results[-1]*100:.2f}%")

