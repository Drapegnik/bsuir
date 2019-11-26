# ml lab1


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 10)
```

### 1. read ex1 data


```python
data1 = np.genfromtxt('data/ex1data1.txt', delimiter=',')
rest = pd.DataFrame(data1, columns=['Population', 'Income'])
rest
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
      <th>Population</th>
      <th>Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.1101</td>
      <td>17.59200</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.5277</td>
      <td>9.13020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.5186</td>
      <td>13.66200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.0032</td>
      <td>11.85400</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.8598</td>
      <td>6.82330</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>5.8707</td>
      <td>7.20290</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5.3054</td>
      <td>1.98690</td>
    </tr>
    <tr>
      <th>94</th>
      <td>8.2934</td>
      <td>0.14454</td>
    </tr>
    <tr>
      <th>95</th>
      <td>13.3940</td>
      <td>9.05510</td>
    </tr>
    <tr>
      <th>96</th>
      <td>5.4369</td>
      <td>0.61705</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 2 columns</p>
</div>



### 2. plot data


```python
def get_plot():
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.ylabel('Income, $10.000')
    plt.xlabel('Population of City, 10.000')
    plt.plot(rest['Population'], rest['Income'], 'rx', markersize=10, label='Training Data')
    return plt

plot = get_plot()
plot.show()
```


![png](out/output_5_0.png)


### 3. cost function J(θ)


```python
# Linear hypothesis function
def h(X, theta):
    return np.dot(X, theta)

# J = compute_cost(X, y, theta)
# computes the cost of using theta as the parameter for linear regression to fit the data points in X and y
def compute_cost(X, y, theta):
    m = y.size
    loss = h(X, theta) - y
    return np.sum(np.square(loss)) / (2. * m)
```


```python
(_, n) = rest.shape
theta = np.zeros((1, n)).T

X1 = rest[['Population']]
X1.insert(0, 'theta_0', 1)
y1 = rest[['Income']]

J = compute_cost(X1, y1, theta)
print(f'theta:\t{theta.ravel()}\nJ:\t{float(J)}')
```

    theta:	[0. 0.]
    J:	32.072733877455676


### 4. gradient descent


```python
# Performs gradient descent to learn theta
def gradient_descent(X, y, theta, alpha=0.01, iterations=1500):
    m = y.size
    J_history = []
    XT = X.T

    for i in range(iterations):
        loss = h(X, theta) - y
        gradient = np.dot(XT, loss) / m
        theta -= alpha * gradient
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

theta, j_history = gradient_descent(X1, y1, theta)
print(f'computed theta: {theta.ravel()}')
```

    computed theta: [-3.63029144  1.16636235]



```python
sample_population = [3, 11, 15, 16, 18.5]
predicted_income = [np.dot([1, x], theta).sum() for x in sample_population]
pd.DataFrame(zip(sample_population, predicted_income), columns=['Sample Population', 'Predicted Income'])
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
      <th>Sample Population</th>
      <th>Predicted Income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.0</td>
      <td>-0.316625</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.0</td>
      <td>9.227581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.0</td>
      <td>13.999684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.0</td>
      <td>15.192709</td>
    </tr>
    <tr>
      <th>4</th>
      <td>18.5</td>
      <td>18.175274</td>
    </tr>
  </tbody>
</table>
</div>




```python
h_values = [np.dot(x, theta).sum() for x in X1.to_numpy()]

plot = get_plot()
plot.plot(rest['Population'], h_values, 'b-', label='Hypothesis')
plot.legend()
plot.show()
```


![png](out/output_12_0.png)


###  5. visualizing J(θ)


```python
# grid coordinates for plotting
xvals = np.linspace(-10, 10, 50)
yvals = np.linspace(-1, 4, 50)
xx, yy = np.meshgrid(xvals, yvals, indexing='xy')
Z = np.zeros((xvals.size, yvals.size))

# calculate Z-values (Cost) based on grid of coefficients
for (i, j), v in np.ndenumerate(Z):
    Z[i, j] = compute_cost(X1, y1, theta=[[xx[i, j]], [yy[i, j]]])
```


```python
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# left plot
CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(*theta, c='r')

# right plot
ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Z.min(), Z.max())
ax2.view_init(elev=15, azim=230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)
```


![png](out/output_15_0.png)


### 6. read ex2 data


```python
data2 = np.genfromtxt('data/ex1data2.txt', delimiter=',')
houses = pd.DataFrame(data2, columns=['Area', 'Bedrooms', 'Price'])
houses
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
      <th>Area</th>
      <th>Bedrooms</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2104.0</td>
      <td>3.0</td>
      <td>399900.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1600.0</td>
      <td>3.0</td>
      <td>329900.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>3.0</td>
      <td>369000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1416.0</td>
      <td>2.0</td>
      <td>232000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000.0</td>
      <td>4.0</td>
      <td>539900.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2567.0</td>
      <td>4.0</td>
      <td>314000.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1200.0</td>
      <td>3.0</td>
      <td>299000.0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>852.0</td>
      <td>2.0</td>
      <td>179900.0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1852.0</td>
      <td>4.0</td>
      <td>299900.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1203.0</td>
      <td>3.0</td>
      <td>239500.0</td>
    </tr>
  </tbody>
</table>
<p>47 rows × 3 columns</p>
</div>



### 7. features normalization


```python
# Normalizes the features in X:
#   returns a normalized version of X where
#   the mean value of each feature is 0 and the standard deviation is 1
def feature_normalization(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (X - mu) / sigma, mu, sigma
```


```python
X2 = houses[['Area', 'Bedrooms']]
X2_norm, mu, sigma = feature_normalization(X2)
X2_norm.describe()
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
      <th>Area</th>
      <th>Bedrooms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.700000e+01</td>
      <td>4.700000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.779483e-17</td>
      <td>2.185013e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.445423e+00</td>
      <td>-2.851859e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-7.155897e-01</td>
      <td>-2.236752e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.417900e-01</td>
      <td>-2.236752e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.376348e-01</td>
      <td>1.090417e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.117292e+00</td>
      <td>2.404508e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
y2 = houses[['Price']]
_, n2 = houses.shape

X2.insert(0, 'theta_0', 1)
X2_norm.insert(0, 'theta_0', 1)
```


```python
t1 = np.zeros((1, n2)).T
t2 = np.zeros((1, n2)).T
(theta1, j_history) = gradient_descent(X2, y2, t1, 0.00000001, 50)
(theta2, j_norm_history) = gradient_descent(X2_norm, y2, t2, 0.1, 50)
print(f'theta1:\t{theta1.ravel()}\ntheta2:\t{theta2.ravel()}')
```

    theta1:	[6.96485826e-02 1.49853133e+02 2.24735775e-01]
    theta2:	[ 3.38658249e+05  1.04127516e+05 -1.72205334e+02]



```python
p1 = plt.plot(range(len(j_history)), j_history, color='black')
p2 = plt.plot(range(len(j_norm_history)), j_norm_history, color='red')
plt.legend((p1[0], p2[0]), ('raw', 'normalized'))
plt.show()
```


![png](out/output_23_0.png)


### 8. multi gradient descent with vectorizing


```python
alpha = 0.01
iterations = 400


(theta_mul, _) = gradient_descent(X2_norm, y2, np.zeros((1, n2)).T, alpha, iterations)
print(f'theta_mul:\t{theta_mul.ravel()}')
```

    theta_mul:	[334302.06399328 100087.11600585   3673.54845093]


### 9. execution time


```python
from timeit import default_timer

iterations = 1000
alpha = 0.02

start = default_timer()
(theta_timer, _) = gradient_descent(X2_norm.to_numpy(), y2.to_numpy(), np.zeros((1, n2)).T, alpha, iterations)
end = default_timer()
print(f'theta_timer:\t{theta_timer.ravel()}\ttime:{end - start}')
```

    theta_timer:	[340412.65900156 110620.78816241  -6639.21215439]	time:0.02559936400211882


### 10. ɑ varying plot


```python
def draw_alphas(iterations):
    alphas = np.linspace(0.1, 0.001, num=7)
    plots = []
    for alpha in alphas:
        (theta, j_hist) = gradient_descent(X2_norm.to_numpy(), y2.to_numpy(), np.zeros((1, n2)).T, alpha, iterations)
        p = plt.plot(range(len(j_hist)), j_hist)
        plots.append(p[0])

    plt.title(f'iterations: {iterations}')
    plt.legend(plots, [f'Alpha: {a:.3f}' for a in alphas])
    plt.show()
```


```python
draw_alphas(30)
draw_alphas(60)
draw_alphas(100)
```


![png](out/output_30_0.png)



![png](out/output_30_1.png)



![png](out/output_30_2.png)


### 11. least squares


```python
# computes the closed-form solution to linear regression using the normal equations
def normal_eqn(X, y):
    XX = np.asmatrix(X)
    XT = XX.T
    return np.array([float(el) for el in ((XT @ XX).I @ XT) @ y])
```


```python
theta_sq = normal_eqn(X2.to_numpy(), y2.to_numpy())
print(f'theta_sq:\t{theta_sq.ravel()}\ntheta_gd:\t{theta_mul.ravel()}')
```

    theta_sq:	[89597.9095428    139.21067402 -8738.01911233]
    theta_gd:	[334302.06399328 100087.11600585   3673.54845093]



```python
AREA = 1890
ROOMS = 4

price_sq = np.array([1, AREA, ROOMS]) @ theta_sq.T
price_gd = (np.array([1, (AREA - mu[0]) / sigma[0], (ROOMS - mu[1]) / sigma[1]]) @ theta_mul)[0]
print(f'price_sq:\t{price_sq}\nprice_gd:\t{price_gd}')
```

    price_sq:	317754.00698679825
    price_gd:	324368.29504053446


### 12. conclusion

В лабараторной работе были рассмотрены случаи линейной и многомерной регресии с помощью методов **градиентного спуска** [#4] а также аналититеского метода **наименьших квадратов** [#11].

В работе использовался язык программирования **Python**, интерактиваня среда разработки **Jupyter** а также библиотеки `numpy`, `pandas` и `matplotlib`

- Как видно из графика [#7] нормализация увеличивает скорость сходимости градиентного спуска.
- В пункте #10 показана зависимость скорости сходимости от параметра ɑ и количества итераций.
- В пункте #11 метод градиентного спуска сравнивается с методом наименьших квадратов.
