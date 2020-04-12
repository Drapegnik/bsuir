# ml lab6


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### 0. load dataset


```python
!chmod +x ./download_data.sh
!./download_data.sh
```

    data/sign-language-mnist already exist


### 1. train / test split


```python
DATASET_DIR = './data/sign-language-mnist'

train_df = pd.read_csv(f'{DATASET_DIR}/sign_mnist_train.csv')
test_df = pd.read_csv(f'{DATASET_DIR}/sign_mnist_test.csv')
```


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
      <th>pixel784</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>107</td>
      <td>118</td>
      <td>127</td>
      <td>134</td>
      <td>139</td>
      <td>143</td>
      <td>146</td>
      <td>150</td>
      <td>153</td>
      <td>...</td>
      <td>207</td>
      <td>207</td>
      <td>207</td>
      <td>207</td>
      <td>206</td>
      <td>206</td>
      <td>206</td>
      <td>204</td>
      <td>203</td>
      <td>202</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>155</td>
      <td>157</td>
      <td>156</td>
      <td>156</td>
      <td>156</td>
      <td>157</td>
      <td>156</td>
      <td>158</td>
      <td>158</td>
      <td>...</td>
      <td>69</td>
      <td>149</td>
      <td>128</td>
      <td>87</td>
      <td>94</td>
      <td>163</td>
      <td>175</td>
      <td>103</td>
      <td>135</td>
      <td>149</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>187</td>
      <td>188</td>
      <td>188</td>
      <td>187</td>
      <td>187</td>
      <td>186</td>
      <td>187</td>
      <td>188</td>
      <td>187</td>
      <td>...</td>
      <td>202</td>
      <td>201</td>
      <td>200</td>
      <td>199</td>
      <td>198</td>
      <td>199</td>
      <td>198</td>
      <td>195</td>
      <td>194</td>
      <td>195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>211</td>
      <td>211</td>
      <td>212</td>
      <td>212</td>
      <td>211</td>
      <td>210</td>
      <td>211</td>
      <td>210</td>
      <td>210</td>
      <td>...</td>
      <td>235</td>
      <td>234</td>
      <td>233</td>
      <td>231</td>
      <td>230</td>
      <td>226</td>
      <td>225</td>
      <td>222</td>
      <td>229</td>
      <td>163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>164</td>
      <td>167</td>
      <td>170</td>
      <td>172</td>
      <td>176</td>
      <td>179</td>
      <td>180</td>
      <td>184</td>
      <td>185</td>
      <td>...</td>
      <td>92</td>
      <td>105</td>
      <td>105</td>
      <td>108</td>
      <td>133</td>
      <td>163</td>
      <td>157</td>
      <td>163</td>
      <td>164</td>
      <td>179</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 785 columns</p>
</div>




```python
labels_count = len(set(train_df.label))
labels_count
```




    24




```python
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)
train_df.shape, val_df.shape, test_df.shape
```




    ((20591, 785), (6864, 785), (7172, 785))




```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

image_width = 28
batch_size = 64

def cook_data(df, shuffle, generator=ImageDataGenerator(rescale=1./255), expand=False):
    new_df = df.reset_index(drop=True)
    labels = new_df.label.values
    images = new_df.drop(columns='label').values.reshape(-1, image_width, image_width, 1)

    if expand:
        images = np.concatenate([images] * 3, axis=-1)
        pad = (2, 2)
        images = np.pad(images, ((0, 0), pad, pad, (0, 0)), constant_values=(0, 0))
    print(images.shape)

    return generator.flow(
        images,
        to_categorical(labels),
        batch_size=batch_size,
        shuffle=shuffle,
    )
```


```python
train = cook_data(train_df, True)
val = cook_data(val_df, False)
test = cook_data(test_df, False)
```

    (20591, 28, 28, 1)
    (6864, 28, 28, 1)
    (7172, 28, 28, 1)


### 2. build network


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Dropout, Flatten

def build_model():
    model = Sequential()

    model.add(Conv2D(32, 3, padding='same', input_shape=(image_width, image_width, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(labels_count + 1, activation='softmax'))

    return model
```


```python
model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])    
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_9 (Conv2D)            (None, 28, 28, 32)        320       
    _________________________________________________________________
    batch_normalization_12 (Batc (None, 28, 28, 32)        128       
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 12, 12, 64)        18496     
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 12, 12, 64)        256       
    _________________________________________________________________
    max_pooling2d_10 (MaxPooling (None, 6, 6, 64)          0         
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 6, 6, 64)          0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 4, 4, 128)         73856     
    _________________________________________________________________
    batch_normalization_14 (Batc (None, 4, 4, 128)         512       
    _________________________________________________________________
    max_pooling2d_11 (MaxPooling (None, 2, 2, 128)         0         
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 2, 2, 128)         0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 512)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 512)               262656    
    _________________________________________________________________
    batch_normalization_15 (Batc (None, 512)               2048      
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 512)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 25)                12825     
    =================================================================
    Total params: 371,097
    Trainable params: 369,625
    Non-trainable params: 1,472
    _________________________________________________________________



```python
from tensorflow.keras.callbacks import EarlyStopping

history = model.fit(
    train,
    validation_data=val,
    epochs=10,
    workers=4,
    callbacks=[EarlyStopping(patience=5)]
)
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train for 322 steps, validate for 108 steps
    Epoch 1/10
    322/322 [==============================] - 6s 20ms/step - loss: 1.2148 - accuracy: 0.6414 - val_loss: 2.2820 - val_accuracy: 0.2599
    Epoch 2/10
    322/322 [==============================] - 6s 18ms/step - loss: 0.2128 - accuracy: 0.9289 - val_loss: 0.0817 - val_accuracy: 0.9774
    Epoch 3/10
    322/322 [==============================] - 6s 18ms/step - loss: 0.0881 - accuracy: 0.9726 - val_loss: 0.0053 - val_accuracy: 0.9996
    Epoch 4/10
    322/322 [==============================] - 6s 18ms/step - loss: 0.0460 - accuracy: 0.9865 - val_loss: 0.0100 - val_accuracy: 0.9990
    Epoch 5/10
    322/322 [==============================] - 6s 18ms/step - loss: 0.0309 - accuracy: 0.9910 - val_loss: 0.0024 - val_accuracy: 0.9997
    Epoch 6/10
    322/322 [==============================] - 6s 20ms/step - loss: 0.0214 - accuracy: 0.9939 - val_loss: 4.4416e-04 - val_accuracy: 1.0000
    Epoch 7/10
    322/322 [==============================] - 6s 20ms/step - loss: 0.0196 - accuracy: 0.9942 - val_loss: 4.2381e-04 - val_accuracy: 1.0000
    Epoch 8/10
    322/322 [==============================] - 6s 20ms/step - loss: 0.0161 - accuracy: 0.9952 - val_loss: 0.0180 - val_accuracy: 0.9937
    Epoch 9/10
    322/322 [==============================] - 6s 20ms/step - loss: 0.0165 - accuracy: 0.9947 - val_loss: 0.0019 - val_accuracy: 0.9994
    Epoch 10/10
    322/322 [==============================] - 6s 20ms/step - loss: 0.0140 - accuracy: 0.9953 - val_loss: 0.0377 - val_accuracy: 0.9872



```python
def evaluate(model, history):
    results = model.evaluate(test)
    print(f'Train accuracy: {history.history["accuracy"][-1]*100:.2f}%')
    print(f'Test accuracy: {results[-1]*100:.2f}%')
```


```python
evaluate(model, history)
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    113/113 [==============================] - 0s 4ms/step - loss: 0.1948 - accuracy: 0.9426
    Train accuracy: 99.53%
    Test accuracy: 94.26%



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


![png](./out/output_18_0.png)


### 2. data augmentation


```python
train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)


train = cook_data(train_df, True, train_generator)
val = cook_data(val_df, False)
test = cook_data(test_df, True)
```

    (20591, 28, 28, 1)
    (6864, 28, 28, 1)
    (7172, 28, 28, 1)



```python
model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
history = model.fit(
    train,
    epochs=10,
    validation_data=test,
    workers=4,
    callbacks=[EarlyStopping(patience=5)]
)
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train for 322 steps, validate for 113 steps
    Epoch 1/10
    322/322 [==============================] - 7s 23ms/step - loss: 2.3919 - accuracy: 0.3370 - val_loss: 4.3232 - val_accuracy: 0.1341
    Epoch 2/10
    322/322 [==============================] - 7s 21ms/step - loss: 1.1566 - accuracy: 0.6132 - val_loss: 0.5382 - val_accuracy: 0.8424
    Epoch 3/10
    322/322 [==============================] - 7s 21ms/step - loss: 0.7449 - accuracy: 0.7421 - val_loss: 0.5061 - val_accuracy: 0.8331
    Epoch 4/10
    322/322 [==============================] - 7s 20ms/step - loss: 0.5581 - accuracy: 0.8062 - val_loss: 0.2685 - val_accuracy: 0.9179
    Epoch 5/10
    322/322 [==============================] - 8s 23ms/step - loss: 0.4268 - accuracy: 0.8529 - val_loss: 0.2150 - val_accuracy: 0.9232
    Epoch 6/10
    322/322 [==============================] - 8s 24ms/step - loss: 0.3549 - accuracy: 0.8792 - val_loss: 0.1810 - val_accuracy: 0.9442
    Epoch 7/10
    322/322 [==============================] - 7s 23ms/step - loss: 0.3017 - accuracy: 0.8945 - val_loss: 0.0509 - val_accuracy: 0.9855
    Epoch 8/10
    322/322 [==============================] - 8s 25ms/step - loss: 0.2673 - accuracy: 0.9106 - val_loss: 0.4881 - val_accuracy: 0.8422
    Epoch 9/10
    322/322 [==============================] - 8s 25ms/step - loss: 0.2370 - accuracy: 0.9196 - val_loss: 0.2368 - val_accuracy: 0.9247
    Epoch 10/10
    322/322 [==============================] - 8s 25ms/step - loss: 0.2123 - accuracy: 0.9259 - val_loss: 0.1069 - val_accuracy: 0.9787



```python
evaluate(model, history)
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    113/113 [==============================] - 0s 4ms/step - loss: 0.1069 - accuracy: 0.9787
    Train accuracy: 92.59%
    Test accuracy: 97.87%



```python
plot(history)
```


![png](./out/output_24_0.png)


> augmentation prevents overfitting

### 4. Try VGG16

> The default input size for this model is `299x299x3`.

> width and height should be no smaller than `32`


```python
train_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)


train = cook_data(train_df, True, train_generator, expand=True)
val = cook_data(val_df, False, ImageDataGenerator(rescale=1./255), expand=True)
test = cook_data(test_df, True, ImageDataGenerator(rescale=1./255), expand=True)
```

    (20591, 32, 32, 3)
    (6864, 32, 32, 3)
    (7172, 32, 32, 3)



```python
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

image_shape = (32, 32, 3)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)

for i, layer in enumerate(base_model.layers):
    layer.trainable = i > 20

last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output

x = GlobalAveragePooling2D()(last_output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(labels_count + 1, activation='softmax')(x)

model = Model(base_model.input, x)

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()
```

    Model: "model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 32, 32, 3)]       0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 32, 32, 64)        1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 32, 32, 64)        36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 16, 16, 64)        0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 16, 16, 128)       73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 16, 16, 128)       147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 8, 8, 128)         0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 8, 8, 256)         295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 8, 8, 256)         590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 8, 8, 256)         590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 4, 4, 256)         0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 4, 4, 512)         1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 4, 4, 512)         2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 4, 4, 512)         2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 2, 2, 512)         0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 2, 2, 512)         2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 2, 2, 512)         2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 2, 2, 512)         2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 1, 1, 512)         0         
    _________________________________________________________________
    global_average_pooling2d (Gl (None, 512)               0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 512)               262656    
    _________________________________________________________________
    dropout_20 (Dropout)         (None, 512)               0         
    _________________________________________________________________
    dense_11 (Dense)             (None, 25)                12825     
    =================================================================
    Total params: 14,990,169
    Trainable params: 275,481
    Non-trainable params: 14,714,688
    _________________________________________________________________



```python
history = model.fit(
    train,
    epochs=20,
    validation_data=val,
    workers=4,
    callbacks=[EarlyStopping(patience=5)]
)
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    Train for 322 steps, validate for 108 steps
    Epoch 1/20
    322/322 [==============================] - 28s 88ms/step - loss: 0.8279 - accuracy: 0.7090 - val_loss: 0.7640 - val_accuracy: 0.7341
    Epoch 2/20
    322/322 [==============================] - 33s 101ms/step - loss: 0.8022 - accuracy: 0.7188 - val_loss: 0.7954 - val_accuracy: 0.7162
    Epoch 3/20
    322/322 [==============================] - 33s 101ms/step - loss: 0.7817 - accuracy: 0.7263 - val_loss: 0.8450 - val_accuracy: 0.6879
    Epoch 4/20
    322/322 [==============================] - 32s 101ms/step - loss: 0.7672 - accuracy: 0.7305 - val_loss: 0.7095 - val_accuracy: 0.7461
    Epoch 5/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.7550 - accuracy: 0.7370 - val_loss: 0.7092 - val_accuracy: 0.7423
    Epoch 6/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.7299 - accuracy: 0.7439 - val_loss: 0.7102 - val_accuracy: 0.7411
    Epoch 7/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.7091 - accuracy: 0.7513 - val_loss: 0.6953 - val_accuracy: 0.7539
    Epoch 8/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.7041 - accuracy: 0.7497 - val_loss: 0.6904 - val_accuracy: 0.7472
    Epoch 9/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6875 - accuracy: 0.7565 - val_loss: 0.6877 - val_accuracy: 0.7602
    Epoch 10/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6805 - accuracy: 0.7607 - val_loss: 0.6721 - val_accuracy: 0.7534
    Epoch 11/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6599 - accuracy: 0.7668 - val_loss: 0.6589 - val_accuracy: 0.7637
    Epoch 12/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6604 - accuracy: 0.7679 - val_loss: 0.6044 - val_accuracy: 0.7854
    Epoch 13/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6488 - accuracy: 0.7710 - val_loss: 0.6662 - val_accuracy: 0.7617
    Epoch 14/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6377 - accuracy: 0.7732 - val_loss: 0.7034 - val_accuracy: 0.7341
    Epoch 15/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6343 - accuracy: 0.7754 - val_loss: 0.6649 - val_accuracy: 0.7526
    Epoch 16/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6295 - accuracy: 0.7733 - val_loss: 0.5959 - val_accuracy: 0.7888
    Epoch 17/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6115 - accuracy: 0.7848 - val_loss: 0.5792 - val_accuracy: 0.7927
    Epoch 18/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6102 - accuracy: 0.7834 - val_loss: 0.5772 - val_accuracy: 0.8016
    Epoch 19/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.6130 - accuracy: 0.7848 - val_loss: 0.5802 - val_accuracy: 0.7837
    Epoch 20/20
    322/322 [==============================] - 33s 102ms/step - loss: 0.5991 - accuracy: 0.7894 - val_loss: 0.5944 - val_accuracy: 0.8033



```python
evaluate(model, history)
```

    WARNING:tensorflow:sample_weight modes were coerced from
      ...
        to  
      ['...']
    113/113 [==============================] - 7s 58ms/step - loss: 0.8935 - accuracy: 0.6735
    Train accuracy: 78.94%
    Test accuracy: 67.35%



```python
plot(history)
```


![png](./out/output_32_0.png)
