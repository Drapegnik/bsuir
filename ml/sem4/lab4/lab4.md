# ml lab4


```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
```

### 0. load mnist data


```python
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)        
```


```python
X_train.shape, X_test.shape
```




    ((60000, 28, 28, 1), (10000, 28, 28, 1))




```python
from random import randrange

def show_images(X, y, shift=0):
    fig, axs = plt.subplots(1, 10, figsize=(20, 2))
    for ax in axs:
        i = randrange(X.shape[0])
        ax.imshow(X[i, :, :, 0])
        ax.set_title(y[i] + shift)
        ax.axis('off')
```


```python
show_images(X_train, y_train)
```


![png](./out/output_6_0.png)


### 1. build network


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Activation

def build_model(image_shape, output_size):
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(output_size))
    model.add(Activation('softmax'))

    return model
```


```python
from keras_tqdm import TQDMNotebookCallback

model = build_model(X_train.shape[1:], 10)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=0,
    callbacks=[TQDMNotebookCallback()]
)
```

    Using TensorFlow backend.



    HBox(children=(FloatProgress(value=0.0, description='Training', max=20.0, style=ProgressStyle(description_widt…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 0', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 2', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 3', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 4', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 5', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 6', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 7', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 8', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 9', max=60000.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 10', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 11', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 12', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 13', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 14', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 15', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 16', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 17', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 18', max=60000.0, style=ProgressStyle(description_w…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 19', max=60000.0, style=ProgressStyle(description_w…






```python
def plot(_history):    
    plt.figure(figsize=(8, 6))
    plt.plot(_history.history['accuracy'], 'r')
    plt.plot(_history.history['val_accuracy'], 'b')
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
```


```python
plot(history)
```


![png](./out/output_11_0.png)


### 2. train on svhn data


```python
import os
import requests
import humanize

DATA_DIR = './data'

def fetch(url):
    filename = os.path.basename(url)
    filepath = f'{DATA_DIR}/{filename}'

    if os.path.exists(filepath):
        return

    r = requests.get(url)
    size = r.headers.get('content-length', 0)
    print(filepath)
    print(f'size:\t{humanize.naturalsize(size)}')

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(filepath, 'wb') as f:
        f.write(r.content)
```


```python
fetch('http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
```

    ./data/train_32x32.mat
    size:	182.0 MB



```python
fetch('http://ufldl.stanford.edu/housenumbers/test_32x32.mat')
```

    ./data/test_32x32.mat
    size:	64.3 MB



```python
import scipy.io

def read_data(filename):
    filepath = f'{DATA_DIR}/{filename}'
    data = scipy.io.loadmat(filepath)
    return np.moveaxis(data['X'], -1, 0), data['y'].flatten() - 1

X_train, y_train = read_data('train_32x32.mat')
X_test, y_test = read_data('test_32x32.mat')
```


```python
show_images(X_train, y_train, 1)
```


![png](./out/output_17_0.png)



```python
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```




    ((73257, 32, 32, 3), (73257,), (26032, 32, 32, 3), (26032,))




```python
model = build_model(X_train.shape[1:], len(np.unique(y_train)))
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 32, 32, 32)        896       
    _________________________________________________________________
    activation_6 (Activation)    (None, 32, 32, 32)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 30, 30, 32)        9248      
    _________________________________________________________________
    activation_7 (Activation)    (None, 30, 30, 32)        0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 15, 15, 32)        0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 15, 15, 64)        18496     
    _________________________________________________________________
    activation_8 (Activation)    (None, 15, 15, 64)        0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 13, 13, 64)        36928     
    _________________________________________________________________
    activation_9 (Activation)    (None, 13, 13, 64)        0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 6, 6, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 2304)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               1180160   
    _________________________________________________________________
    activation_10 (Activation)   (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                5130      
    _________________________________________________________________
    activation_11 (Activation)   (None, 10)                0         
    =================================================================
    Total params: 1,250,858
    Trainable params: 1,250,858
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=0,
    callbacks=[TQDMNotebookCallback()]
)
```


    HBox(children=(FloatProgress(value=0.0, description='Training', max=10.0, style=ProgressStyle(description_widt…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 0', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 2', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 3', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 4', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 5', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 6', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 7', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 8', max=73257.0, style=ProgressStyle(description_wi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 9', max=73257.0, style=ProgressStyle(description_wi…






```python
plot(history)
```


![png](./out/output_22_0.png)



```python
loss, acc = model.evaluate(X_test, y_test, batch_size=128)
```

    26032/26032 [==============================] - 3s 98us/sample - loss: 0.8024 - accuracy: 0.8846



```python
print(f'{acc * 100:.2f}%')
```

    88.46%



```python
predictions = model.predict_classes(X_test)
```


```python
fig, axs = plt.subplots(1, 10, figsize=(20, 2))
for ax in axs:
    i = randrange(X_test.shape[0])
    ax.imshow(X_test[i, :, :, 0])
    ax.set_title(f'Pred: {predictions[i] + 1}, True: {y_test[i] + 1}')
    ax.axis('off')
```


![png](./out/output_26_0.png)


### 3. save model


```python
model.save(f'{DATA_DIR}/svhn_model.h5', save_format='h5')
```

### 4. telegram bot


```python
from tensorflow.keras.models import load_model

model = load_model('data/svhn_model.h5')

def download_image(file_path):
    url = f'{BASE_FILES_URL}/{file_path}'
    with request.urlopen(url) as res:
        image = np.asarray(bytearray(res.read()), dtype='uint8')
        image = cv.imdecode(image, cv.IMREAD_COLOR)
        image = cv.resize(image, IMAGE_SIZE)
        return image


def predict(image):
    X = np.asarray(image).astype('float32')
    X = X.reshape(1, 32, 32, 3)
    predictions = model.predict_classes(X)
    return predictions[0] + 1
```

<table>
<tr>
    <td><img src="./out/IMG_1777.jpg" width="320px"/></td>
    <td><img src="./out/IMG_1778.jpg" width="320px"/></td>
    <td><img src="./out/IMG_1779.jpg" width="320px"/></td>
</tr>
</table>

---------------------------
