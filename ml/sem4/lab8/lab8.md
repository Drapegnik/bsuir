](./out/output# ml lab8


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

### 0. download dataset


```python
!./download_data.sh
```

    data/sunspots does not exist, download:
    Downloading sunspots.zip to data
      0%|                                               | 0.00/22.4k [00:00<?, ?B/s]
    100%|███████████████████████████████████████| 22.4k/22.4k [00:00<00:00, 635kB/s]
    Archive:  data/sunspots.zip
      inflating: data/sunspots/Sunspots.csv  


### 1. plot data & compute metrics


```python
DATASET_DIR = './data/sunspots'

data = pd.read_csv(f'{DATASET_DIR}/Sunspots.csv', parse_dates=['Date'], index_col=['Date'])
data.rename(columns = {
    'Unnamed: 0': 'index',
    'Monthly Mean Total Sunspot Number': 'sunspots'
}, inplace = True)
data.head()
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
      <th>index</th>
      <th>sunspots</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1749-01-31</th>
      <td>0</td>
      <td>96.7</td>
    </tr>
    <tr>
      <th>1749-02-28</th>
      <td>1</td>
      <td>104.3</td>
    </tr>
    <tr>
      <th>1749-03-31</th>
      <td>2</td>
      <td>116.7</td>
    </tr>
    <tr>
      <th>1749-04-30</th>
      <td>3</td>
      <td>92.8</td>
    </tr>
    <tr>
      <th>1749-05-31</th>
      <td>4</td>
      <td>141.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 3252 entries, 1749-01-31 to 2019-12-31
    Data columns (total 2 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   index     3252 non-null   int64  
     1   sunspots  3252 non-null   float64
    dtypes: float64(1), int64(1)
    memory usage: 76.2 KB



```python
from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose

rcParams.update({
    'figure.figsize': (20, 16),
    'font.size': 16,
})

idx = pd.Index(sm.tsa.datetools.dates_from_range('1749', '2017'))

result = seasonal_decompose(data.sunspots[idx], model='additive')
result.plot();
```


![png](./out/output_7_0.png)



```python
from pandas.plotting import autocorrelation_plot

rcParams.update({'figure.figsize': (20, 8)})

autocorrelation_plot(data.sunspots)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f16ccbc9a10>




![png](./out/output_8_1.png)



```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data.sunspots, lags=300);
plot_pacf(data.sunspots, lags=50);
```


![png](./out/output_9_0.png)



![png](./out/output_9_1.png)



```python
def trendline(index, data, order=1):
    coeffs = np.polyfit(index, data, order)
    slope = coeffs[-2]
    return float(slope)

trendline(range(data.sunspots.count()), data.sunspots)
```




    0.0031718163496368117



> slope is a zero value: **No trend**

### 2. train / test split


```python
import statsmodels.api as sm

train = pd.Index(sm.tsa.datetools.dates_from_range('1749', '2008'))
test = pd.Index(sm.tsa.datetools.dates_from_range('1990', '2012'))

train[0], train[-1], test[0], test[-1]
```




    (Timestamp('1749-12-31 00:00:00'),
     Timestamp('2008-12-31 00:00:00'),
     Timestamp('1990-12-31 00:00:00'),
     Timestamp('2012-12-31 00:00:00'))



### 3. forecast with ARIMA

The parameters of the ARIMA model are defined as follows:

- `p`: The number of lag observations included in the model, also called the lag order.
- `d`: The number of times that the raw observations are differenced, also called the degree of differencing.
- `q`: The size of the moving average window, also called the order of moving average.


```python
%%time

model = sm.tsa.ARIMA(data.sunspots[train], order=(10, 1, 0)).fit()  
model.summary()
```

    /home/drapegnik/.pyenv/versions/3.7.6/envs/bsuir/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:162: ValueWarning: No frequency information was provided, so inferred frequency A-DEC will be used.
      % freq, ValueWarning)
    /home/drapegnik/.pyenv/versions/3.7.6/envs/bsuir/lib/python3.7/site-packages/statsmodels/tsa/base/tsa_model.py:162: ValueWarning: No frequency information was provided, so inferred frequency A-DEC will be used.
      % freq, ValueWarning)


    CPU times: user 5.53 s, sys: 8.36 s, total: 13.9 s
    Wall time: 1.79 s





<table class="simpletable">
<caption>ARIMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>    <td>D.sunspots</td>    <th>  No. Observations:  </th>    <td>259</td>   
</tr>
<tr>
  <th>Model:</th>          <td>ARIMA(10, 1, 0)</td> <th>  Log Likelihood     </th> <td>-1340.292</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>  <td>42.551</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Sun, 19 Apr 2020</td> <th>  AIC                </th> <td>2704.583</td>
</tr>
<tr>
  <th>Time:</th>              <td>15:44:48</td>     <th>  BIC                </th> <td>2747.265</td>
</tr>
<tr>
  <th>Sample:</th>           <td>12-31-1750</td>    <th>  HQIC               </th> <td>2721.744</td>
</tr>
<tr>
  <th></th>                 <td>- 12-31-2008</td>   <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>             <td>    0.0073</td> <td>    0.647</td> <td>    0.011</td> <td> 0.991</td> <td>   -1.261</td> <td>    1.275</td>
</tr>
<tr>
  <th>ar.L1.D.sunspots</th>  <td>   -0.2378</td> <td>    0.062</td> <td>   -3.830</td> <td> 0.000</td> <td>   -0.360</td> <td>   -0.116</td>
</tr>
<tr>
  <th>ar.L2.D.sunspots</th>  <td>   -0.3383</td> <td>    0.063</td> <td>   -5.394</td> <td> 0.000</td> <td>   -0.461</td> <td>   -0.215</td>
</tr>
<tr>
  <th>ar.L3.D.sunspots</th>  <td>   -0.3577</td> <td>    0.063</td> <td>   -5.637</td> <td> 0.000</td> <td>   -0.482</td> <td>   -0.233</td>
</tr>
<tr>
  <th>ar.L4.D.sunspots</th>  <td>   -0.3664</td> <td>    0.062</td> <td>   -5.897</td> <td> 0.000</td> <td>   -0.488</td> <td>   -0.245</td>
</tr>
<tr>
  <th>ar.L5.D.sunspots</th>  <td>   -0.4772</td> <td>    0.062</td> <td>   -7.694</td> <td> 0.000</td> <td>   -0.599</td> <td>   -0.356</td>
</tr>
<tr>
  <th>ar.L6.D.sunspots</th>  <td>   -0.3655</td> <td>    0.062</td> <td>   -5.903</td> <td> 0.000</td> <td>   -0.487</td> <td>   -0.244</td>
</tr>
<tr>
  <th>ar.L7.D.sunspots</th>  <td>   -0.4126</td> <td>    0.062</td> <td>   -6.641</td> <td> 0.000</td> <td>   -0.534</td> <td>   -0.291</td>
</tr>
<tr>
  <th>ar.L8.D.sunspots</th>  <td>   -0.3039</td> <td>    0.063</td> <td>   -4.797</td> <td> 0.000</td> <td>   -0.428</td> <td>   -0.180</td>
</tr>
<tr>
  <th>ar.L9.D.sunspots</th>  <td>   -0.2025</td> <td>    0.063</td> <td>   -3.232</td> <td> 0.001</td> <td>   -0.325</td> <td>   -0.080</td>
</tr>
<tr>
  <th>ar.L10.D.sunspots</th> <td>   -0.0884</td> <td>    0.062</td> <td>   -1.420</td> <td> 0.156</td> <td>   -0.210</td> <td>    0.034</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>    <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th>  <td>           0.8570</td> <td>          -0.5736j</td> <td>           1.0312</td> <td>          -0.0939</td>
</tr>
<tr>
  <th>AR.2</th>  <td>           0.8570</td> <td>          +0.5736j</td> <td>           1.0312</td> <td>           0.0939</td>
</tr>
<tr>
  <th>AR.3</th>  <td>           0.4850</td> <td>          -1.1203j</td> <td>           1.2208</td> <td>          -0.1850</td>
</tr>
<tr>
  <th>AR.4</th>  <td>           0.4850</td> <td>          +1.1203j</td> <td>           1.2208</td> <td>           0.1850</td>
</tr>
<tr>
  <th>AR.5</th>  <td>          -0.2311</td> <td>          -1.3342j</td> <td>           1.3541</td> <td>          -0.2773</td>
</tr>
<tr>
  <th>AR.6</th>  <td>          -0.2311</td> <td>          +1.3342j</td> <td>           1.3541</td> <td>           0.2773</td>
</tr>
<tr>
  <th>AR.7</th>  <td>          -0.8153</td> <td>          -1.0840j</td> <td>           1.3564</td> <td>          -0.3526</td>
</tr>
<tr>
  <th>AR.8</th>  <td>          -0.8153</td> <td>          +1.0840j</td> <td>           1.3564</td> <td>           0.3526</td>
</tr>
<tr>
  <th>AR.9</th>  <td>          -1.4411</td> <td>          -0.1980j</td> <td>           1.4547</td> <td>          -0.4783</td>
</tr>
<tr>
  <th>AR.10</th> <td>          -1.4411</td> <td>          +0.1980j</td> <td>           1.4547</td> <td>           0.4783</td>
</tr>
</table>




```python
model.plot_predict('1990', '2012', dynamic=True);
```


![png](./out/output_17_0.png)


### 4. build rnn


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

TIME_PERIOD = 12

def build_model(input_shape):
    model = Sequential()
    model.add(GRU(TIME_PERIOD, input_shape=input_shape, return_sequences=True))
    model.add(GRU(TIME_PERIOD))
    model.add(Dense(1))
    return model
```


```python
def normalize(df):
    mean = df.sunspots.mean()
    std = df.sunspots.std()
    df['sunspots_norm'] = (df.sunspots - mean) / std
    return df

normalize(data)
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
      <th>index</th>
      <th>sunspots</th>
      <th>sunspots_norm</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1749-01-31</th>
      <td>0</td>
      <td>96.7</td>
      <td>0.215566</td>
    </tr>
    <tr>
      <th>1749-02-28</th>
      <td>1</td>
      <td>104.3</td>
      <td>0.327553</td>
    </tr>
    <tr>
      <th>1749-03-31</th>
      <td>2</td>
      <td>116.7</td>
      <td>0.510270</td>
    </tr>
    <tr>
      <th>1749-04-30</th>
      <td>3</td>
      <td>92.8</td>
      <td>0.158098</td>
    </tr>
    <tr>
      <th>1749-05-31</th>
      <td>4</td>
      <td>141.7</td>
      <td>0.878649</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2019-08-31</th>
      <td>3247</td>
      <td>0.5</td>
      <td>-1.201960</td>
    </tr>
    <tr>
      <th>2019-09-30</th>
      <td>3248</td>
      <td>1.1</td>
      <td>-1.193119</td>
    </tr>
    <tr>
      <th>2019-10-31</th>
      <td>3249</td>
      <td>0.4</td>
      <td>-1.203433</td>
    </tr>
    <tr>
      <th>2019-11-30</th>
      <td>3250</td>
      <td>0.5</td>
      <td>-1.201960</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>3251</td>
      <td>1.6</td>
      <td>-1.185751</td>
    </tr>
  </tbody>
</table>
<p>3252 rows × 3 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.3, shuffle=False)
test, val = train_test_split(test, test_size=0.5, shuffle=False)

train.shape, test.shape, val.shape
```




    ((2276, 3), (488, 3), (488, 3))




```python
import tensorflow as tf

BATCH_SIZE = 256

def get_dataset(df, time_period=TIME_PERIOD, shuffle=False, batch_size=BATCH_SIZE, raw=False):
    X = []
    y = []

    for i in range(time_period, len(df)):
        prev_v = np.array(df['sunspots_norm'][i-time_period:i])
        next_v = df['sunspots_norm'][i]

        X.append(prev_v)
        y.append(next_v)

    X = np.array(X)
    X = np.expand_dims(X, axis=-1)
    y = np.array(y)

    if raw:
        return X, y

    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size).repeat()
    return ds
```


```python
train_ds = get_dataset(data, shuffle=True)
test_ds = get_dataset(data)
```


```python
input_shape = train_ds.element_spec[0].shape[1:]
input_shape
```




    TensorShape([12, 1])




```python
model = build_model(input_shape)
model.compile(loss='mse', optimizer='adam')
model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    gru_9 (GRU)                  (None, 12, 12)            540       
    _________________________________________________________________
    gru_10 (GRU)                 (None, 12)                936       
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 13        
    =================================================================
    Total params: 1,489
    Trainable params: 1,489
    Non-trainable params: 0
    _________________________________________________________________



```python
from tensorflow.keras.callbacks import EarlyStopping

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    steps_per_epoch=100,
    validation_steps=30,
    callbacks=[EarlyStopping(patience=5)]
)
```

    Train for 100 steps, validate for 30 steps
    Epoch 1/10
    100/100 [==============================] - 2s 23ms/step - loss: 0.2556 - val_loss: 0.1576
    Epoch 2/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1551 - val_loss: 0.1445
    Epoch 3/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1448 - val_loss: 0.1378
    Epoch 4/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1399 - val_loss: 0.1346
    Epoch 5/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1374 - val_loss: 0.1334
    Epoch 6/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1371 - val_loss: 0.1322
    Epoch 7/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1363 - val_loss: 0.1318
    Epoch 8/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1352 - val_loss: 0.1318
    Epoch 9/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1359 - val_loss: 0.1316
    Epoch 10/10
    100/100 [==============================] - 1s 7ms/step - loss: 0.1356 - val_loss: 0.1314



```python
X_test, y_test = get_dataset(test, raw=True)
X_val, y_val = get_dataset(val, raw=True)
```


```python
test_loss = model.evaluate(X_test, y_test)
val_loss = model.evaluate(X_val, y_val)

print(f'Test loss:\t{test_loss:.2%}')
print(f'Validation loss:\t{val_loss:.2%}')
```

    476/476 [==============================] - 0s 657us/sample - loss: 0.1466
    476/476 [==============================] - 0s 43us/sample - loss: 0.1146
    Test loss:	14.66%
    Validation loss:	11.46%



```python
mean = data.sunspots.mean()
std = data.sunspots.std()

def predict(X):
    shift = [None] * TIME_PERIOD
    predict = mean + model.predict(X) * std
    return shift + list(predict.flatten())
```


```python
HISTORY_FROM = '1917'
history = data.loc[HISTORY_FROM:]
train_size = len(train.loc[HISTORY_FROM:])
train_with_test_size = train_size + len(test)
full_size = train_with_test_size + len(val)

plt.figure(figsize=(20, 4))
plt.plot(np.arange(full_size), history.sunspots, label='History')
plt.plot(np.arange(train_size, train_with_test_size), predict(X_test), label='Test Forecast')
plt.plot(np.arange(train_with_test_size, full_size), predict(X_val), label='Val Forecast')
plt.legend()
plt.show()
```


![png](./out/output_30_0.png)
