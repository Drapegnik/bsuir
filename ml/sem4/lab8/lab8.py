#!/usr/bin/env python
# coding: utf-8

# # ml lab8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 0. download dataset

# In[2]:


get_ipython().system("./download_data.sh")


# ### 1. plot data & compute metrics

# In[47]:


DATASET_DIR = "./data/sunspots"

data = pd.read_csv(
    f"{DATASET_DIR}/Sunspots.csv", parse_dates=["Date"], index_col=["Date"]
)
data.rename(
    columns={"Unnamed: 0": "index", "Monthly Mean Total Sunspot Number": "sunspots"},
    inplace=True,
)
data.head()


# In[154]:


data.info()


# In[177]:


from pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose

rcParams.update(
    {"figure.figsize": (20, 16), "font.size": 16,}
)

idx = pd.Index(sm.tsa.datetools.dates_from_range("1749", "2017"))

result = seasonal_decompose(data.sunspots[idx], model="additive")
result.plot()


# In[190]:


from pandas.plotting import autocorrelation_plot

rcParams.update({"figure.figsize": (20, 8)})

autocorrelation_plot(data.sunspots)


# In[250]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data.sunspots, lags=300)
plot_pacf(data.sunspots, lags=50)


# In[186]:


def trendline(index, data, order=1):
    coeffs = np.polyfit(index, data, order)
    slope = coeffs[-2]
    return float(slope)


trendline(range(data.sunspots.count()), data.sunspots)


# > slope is a zero value: **No trend**

# ### 2. train / test split

# In[283]:


import statsmodels.api as sm

train = pd.Index(sm.tsa.datetools.dates_from_range("1749", "2008"))
test = pd.Index(sm.tsa.datetools.dates_from_range("1990", "2012"))

train[0], train[-1], test[0], test[-1]


# ### 3. forecast with ARIMA

# The parameters of the ARIMA model are defined as follows:
#
# - `p`: The number of lag observations included in the model, also called the lag order.
# - `d`: The number of times that the raw observations are differenced, also called the degree of differencing.
# - `q`: The size of the moving average window, also called the order of moving average.

# In[284]:


get_ipython().run_cell_magic(
    "time",
    "",
    "\nmodel = sm.tsa.ARIMA(data.sunspots[train], order=(10, 1, 0)).fit()  \nmodel.summary()",
)


# In[285]:


model.plot_predict("1990", "2012", dynamic=True)


# ### 4. build rnn

# In[329]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

TIME_PERIOD = 12


def build_model(input_shape):
    model = Sequential()
    model.add(GRU(TIME_PERIOD, input_shape=input_shape, return_sequences=True))
    model.add(GRU(TIME_PERIOD))
    model.add(Dense(1))
    return model


# In[330]:


def normalize(df):
    mean = df.sunspots.mean()
    std = df.sunspots.std()
    df["sunspots_norm"] = (df.sunspots - mean) / std
    return df


normalize(data)


# In[404]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.3, shuffle=False)
test, val = train_test_split(test, test_size=0.5, shuffle=False)

train.shape, test.shape, val.shape


# In[405]:


import tensorflow as tf

BATCH_SIZE = 256


def get_dataset(
    df, time_period=TIME_PERIOD, shuffle=False, batch_size=BATCH_SIZE, raw=False
):
    X = []
    y = []

    for i in range(time_period, len(df)):
        prev_v = np.array(df["sunspots_norm"][i - time_period : i])
        next_v = df["sunspots_norm"][i]

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


# In[406]:


train_ds = get_dataset(data, shuffle=True)
test_ds = get_dataset(data)


# In[407]:


input_shape = train_ds.element_spec[0].shape[1:]
input_shape


# In[408]:


model = build_model(input_shape)
model.compile(loss="mse", optimizer="adam")
model.summary()


# In[409]:


from tensorflow.keras.callbacks import EarlyStopping

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    steps_per_epoch=100,
    validation_steps=30,
    callbacks=[EarlyStopping(patience=5)],
)


# In[410]:


X_test, y_test = get_dataset(test, raw=True)
X_val, y_val = get_dataset(val, raw=True)


# In[411]:


test_loss = model.evaluate(X_test, y_test)
val_loss = model.evaluate(X_val, y_val)

print(f"Test loss:\t{test_loss:.2%}")
print(f"Validation loss:\t{val_loss:.2%}")


# In[412]:


mean = data.sunspots.mean()
std = data.sunspots.std()


def predict(X):
    shift = [None] * TIME_PERIOD
    predict = mean + model.predict(X) * std
    return shift + list(predict.flatten())


# In[431]:


HISTORY_FROM = "1917"
history = data.loc[HISTORY_FROM:]
train_size = len(train.loc[HISTORY_FROM:])
train_with_test_size = train_size + len(test)
full_size = train_with_test_size + len(val)

plt.figure(figsize=(20, 4))
plt.plot(np.arange(full_size), history.sunspots, label="History")
plt.plot(
    np.arange(train_size, train_with_test_size), predict(X_test), label="Test Forecast"
)
plt.plot(
    np.arange(train_with_test_size, full_size), predict(X_val), label="Val Forecast"
)
plt.legend()
plt.show()
