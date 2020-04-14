#!/usr/bin/env python
# coding: utf-8

# # ml lab7

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 1. load dataset & prepare data

# In[57]:


import tensorflow as tf
import tensorflow_datasets as tfds

DATASET_NAME = "imdb_reviews/subwords8k"

data, info = tfds.load(DATASET_NAME, as_supervised=True, with_info=True)

print(info.homepage)

data


# In[61]:


encoder = info.features["text"].encoder
train = data["train"]
test = data["test"]
encoder


# In[118]:


for train_example, train_label in train.take(10):
    encoded = train_example[:10].numpy()
    sentense = encoder.decode(encoded)
    label = "👍" if train_label.numpy() else "😥"
    print(f"[{label}] {sentense.ljust(50)}\t", encoded)


# In[96]:


BUFFER_SIZE = 10000
BATCH_SIZE = 64


def cook_data(ds):
    return ds.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], []))


# In[63]:


train_dataset = cook_data(train)
test_dataset = cook_data(test)


# ### 2. build lstm network

# In[217]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


def build_model(emb=Embedding(encoder.vocab_size, 64)):
    model = Sequential()

    model.add(emb)
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    return model


# In[71]:


model = build_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# In[72]:


from tensorflow.keras.callbacks import EarlyStopping

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    validation_steps=20,
    epochs=10,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[75]:


def evaluate(model, history):
    results = model.evaluate(test_dataset)
    print(f'\nTrain accuracy: {history.history["accuracy"][-1]*100:.2f}%')
    print(f"Test accuracy: {results[-1]*100:.2f}%")


# In[76]:


evaluate(model, history)


# In[77]:


def plot(_history):
    plt.figure(figsize=(8, 6))
    plt.plot(_history.history["accuracy"], "r")
    plt.plot(_history.history["val_accuracy"], "b")
    plt.legend(["Training", "Validation"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")


# In[78]:


plot(history)


# ### 3. glove embeddings

# In[158]:


get_ipython().system("./download_data.sh")


# In[213]:


DATA_DIR = "data"

EMBEDDING_DIM = 200
VOCAB_SIZE = encoder.vocab_size

GLOVE_NAME = f"glove.6B.{EMBEDDING_DIM}d.txt"
GLOVE_PATH = f"{DATA_DIR}/glove-global-vectors-for-word-representation/{GLOVE_NAME}"


# In[169]:


embeddings_index = {}

with open(GLOVE_PATH, "r") as glove_file:
    for line in glove_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors.")


# In[209]:


def clean(s):
    return s.lower().replace("_", "")


# In[214]:


embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
fails = 0

for i, word in enumerate(encoder.subwords, 1):
    embedding_vector = embeddings_index.get(clean(word))
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        fails += 1


# In[215]:


print(f"Found embeddings for {(VOCAB_SIZE - fails) / VOCAB_SIZE:.2%} of vocab")


# In[218]:


model = build_model(
    Embedding(
        encoder.vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False
    )
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# In[222]:


history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    validation_steps=20,
    epochs=10,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[223]:


evaluate(model, history)


# In[224]:


plot(history)


# ### 4. more layers with gru

# In[230]:


from tensorflow.keras.layers import GRU


def build_model():
    model = Sequential()

    model.add(Embedding(VOCAB_SIZE, 100))
    model.add(Bidirectional(GRU(64, return_sequences=True), merge_mode="concat"))
    model.add(Bidirectional(GRU(64), merge_mode="concat"))
    model.add(Dense(64, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    return model


# In[231]:


model = build_model()
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# In[238]:


history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    validation_steps=20,
    epochs=5,
    workers=4,
    callbacks=[EarlyStopping(patience=5)],
)


# In[239]:


evaluate(model, history)


# In[240]:


plot(history)
