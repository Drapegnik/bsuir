#!/usr/bin/env python
# coding: utf-8

# # ml lab6

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 0. load dataset

# In[2]:


get_ipython().system("chmod +x ./download_data.sh ")
get_ipython().system("./download_data.sh")


# ### 1. train / test split

# In[20]:


DATASET_DIR = "./data/sign-language-mnist"

train_df = pd.read_csv(f"{DATASET_DIR}/sign_mnist_train.csv")
test_df = pd.read_csv(f"{DATASET_DIR}/sign_mnist_test.csv")


# In[21]:


train_df.head()


# In[59]:


labels_count = len(set(train_df.label))
labels_count


# In[23]:


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)
train_df.shape, val_df.shape, test_df.shape


# In[147]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

image_width = 28
batch_size = 64


def cook_data(
    df, shuffle, generator=ImageDataGenerator(rescale=1.0 / 255), expand=False
):
    new_df = df.reset_index(drop=True)
    labels = new_df.label.values
    images = new_df.drop(columns="label").values.reshape(
        -1, image_width, image_width, 1
    )

    if expand:
        images = np.concatenate([images] * 3, axis=-1)
        pad = (2, 2)
        images = np.pad(images, ((0, 0), pad, pad, (0, 0)), constant_values=(0, 0))
    print(images.shape)

    return generator.flow(
        images, to_categorical(labels), batch_size=batch_size, shuffle=shuffle,
    )


# In[148]:


train = cook_data(train_df, True)
val = cook_data(val_df, False)
test = cook_data(test_df, False)


# ### 2. build network

# In[89]:


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

    model.add(Conv2D(32, 3, padding="same", input_shape=(image_width, image_width, 1)))
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
    model.add(Dense(labels_count + 1, activation="softmax"))

    return model


# In[90]:


model = build_model()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# In[91]:


from tensorflow.keras.callbacks import EarlyStopping

history = model.fit(
    train,
    validation_data=val,
    epochs=10,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[98]:


def evaluate(model, history):
    results = model.evaluate(test)
    print(f'Train accuracy: {history.history["accuracy"][-1]*100:.2f}%')
    print(f"Test accuracy: {results[-1]*100:.2f}%")


# In[99]:


evaluate(model, history)


# In[100]:


def plot(_history):
    plt.figure(figsize=(8, 6))
    plt.plot(_history.history["accuracy"], "r")
    plt.plot(_history.history["val_accuracy"], "b")
    plt.legend(["Training", "Validation"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")


# In[106]:


plot(history)


# ### 2. data augmentation

# In[150]:


train_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
)


train = cook_data(train_df, True, train_generator)
val = cook_data(val_df, False)
test = cook_data(test_df, True)


# In[151]:


model = build_model()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[152]:


history = model.fit(
    train,
    epochs=10,
    validation_data=test,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[153]:


evaluate(model, history)


# In[154]:


plot(history)


# > augmentation prevents overfitting

# ### 4. Try VGG16

# > The default input size for this model is `299x299x3`.
#
# > width and height should be no smaller than `32`

# In[155]:


train_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
)


train = cook_data(train_df, True, train_generator, expand=True)
val = cook_data(val_df, False, ImageDataGenerator(rescale=1.0 / 255), expand=True)
test = cook_data(test_df, True, ImageDataGenerator(rescale=1.0 / 255), expand=True)


# In[157]:


from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

image_shape = (32, 32, 3)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=image_shape)

for i, layer in enumerate(base_model.layers):
    layer.trainable = i > 20

last_layer = base_model.get_layer("block5_pool")
last_output = last_layer.output

x = GlobalAveragePooling2D()(last_output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(labels_count + 1, activation="softmax")(x)

model = Model(base_model.input, x)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()


# In[159]:


history = model.fit(
    train,
    epochs=20,
    validation_data=val,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[160]:


evaluate(model, history)


# In[161]:


plot(history)
