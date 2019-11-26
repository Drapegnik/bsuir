# ml lab7


```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
```

### 1. read `ex7data1.mat` data


```python
data = scipy.io.loadmat('data/ex7data1.mat')
X = data['X']
X.shape
```




    (300, 2)



### 2. plot data


```python
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b', s=50)
plt.show()
```


![png](out/output_5_0.png)


### 3-4. data covariance matrix + eigenvector


```python
def pca(X):
    m = X.shape[0]
    sigma = 1 / m * X.T.dot(X)
    U, S, V = np.linalg.svd(sigma)
    return U, S

def feature_normalization(X):
    norm = (X - X.mean(axis=0)) / X.std(axis=0)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return norm, mu, sigma

X_norm, mu, sig = feature_normalization(X)
U, S = pca(X_norm)

print(f'Eigen Vector:\t{U[:,0].ravel()}')
```

    Eigen Vector:	[-0.70710678  0.70710678]


### 5. plot eigenvector


```python
plt.figure(figsize=(12, 8))
plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors='b', s=50)

plt.arrow(
    mu[0],
    mu[1],
    1.5 * S[0]*U[0, 0],
    1.5 * S[0]*U[1, 0],
    head_width=0.25,
    head_length=0.2,
    fc='r',
    ec='r',
    lw=2,
    zorder=1000,
)
plt.arrow(
    mu[0],
    mu[1],
    1.5 * S[1]*U[0, 1],
    1.5 * S[1]*U[1, 1],
    head_width=0.25,
    head_length=0.2,
    fc='k',
    ec='k',
    lw=2,
    zorder=1000
)

plt.annotate('First Principal Component', xy=(8.5, 0.3), xytext=(6.3, 0.3), arrowprops={'arrowstyle': '->', 'lw': 1.5, 'ec': 'r'}, va='center')
plt.annotate('Second Principal Component', xy=(8.5, 0.1), xytext=(6.1, 0.1), arrowprops={'arrowstyle': '->', 'lw': 1.5}, va='center')
plt.show()
```


![png](out/output_9_0.png)


# 6.  dimensionality reduction


```python
def project_data(X, U, K):
    return X.dot(U[:, :K])
```

### 7. reconstructing an approximation of the data


```python
def recover_data(Z, U, K):
    return Z.dot(U[:, :K].T)
```

### 8. projections plot


```python
Z = project_data(X_norm, U, 1)
X_rec = recover_data(Z, U, 1)
X_rec = X_rec * sig + mu

print(f'First Approximation:\t{X_rec[0].ravel()}')
```

    First Approximation:	[1.83735239 4.60343487]



```python
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_aspect('equal')
ax.grid(False)

ax.plot(X[:, 0], X[:, 1], 'bo', ms=10, mfc='none', label='Original Data Points')
ax.plot(X_rec[:, 0], X_rec[:, 1], 'ro', ms=10, mfc='none', label='PCA Reduced Data Points')

for xnorm, xrec in zip(X, X_rec):
    ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], 'k--', lw=0.5)

ax.set_title('Example Dataset: Reduced Dimension Points Shown', fontsize=14)
ax.set_xlabel('x1 [Feature Normalized]', fontsize=14)
ax.set_ylabel('x2 [Feature Normalized]', fontsize=14)
plt.legend()
plt.show()
```


![png](out/output_16_0.png)


### 9. read `ex7faces.mat`


```python
data = scipy.io.loadmat('data/ex7faces.mat')
X = data['X']

X.shape
```




    (5000, 1024)



### 10. show 10 random faces


```python
def show_faces(X, title):
    m, n = X.shape

    dim = int(np.sqrt(m))

    fig, axs = plt.subplots(dim, dim, figsize=(8, 8))
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    example_width = int(np.round(np.sqrt(n)))
    example_height = int(n / example_width)
    for i, ax in enumerate(axs.flat):
        ax.axis('off')
        w = int(np.sqrt(n))
        ax.imshow(X[i].reshape(example_height, example_width, order='F'), cmap='gray')

    plt.title(title, loc='right', pad=400)
    plt.show()

show_faces(X[100:200, :], 'Random 100 Faces')
```


![png](out/output_20_0.png)


### 11. get eigenvectors


```python
X_norm, mu, sigma = feature_normalization(X)
U, S = pca(X_norm)
```

### 12. show 36 main components


```python
show_faces(U[:, :36].T, '36 main components')
```


![png](out/output_24_0.png)


### 13. show 100 main components


```python
show_faces(U[:, :100].T, '100 main components')
```


![png](out/output_26_0.png)


> Изображения размываются


```python
K = 100
Z = project_data(X_norm, U, K)
X_rec = recover_data(Z, U, K)
```


```python
show_faces(X_norm[100:200, :], 'Original Data')
```


![png](out/output_29_0.png)



```python
show_faces(X_rec[100:200, :], 'Reconstructed Data')
```


![png](out/output_30_0.png)


### 16-18. `2D` & `3D` visualize compressed image from `lab6`


```python
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

from k_means import find_best_k_means

lena = mpimg.imread('../lab6/data/lena.png')
lena = lena[:, :, :3]

X = np.reshape(lena, newshape=(-1, 3))
K = 16

centroids, idx, _ = find_best_k_means(X, K)
X_rec = centroids[idx]
X_rec = X_rec.reshape(-1, lena.shape[1], 3)
```

    > [0]: k-means improved with distortion: 0.004678657518525142
    > [1]: k-means improved with distortion: 0.004394731760839932
    > [2]: k-means improved with distortion: 0.004112903683562763
    > [7]: k-means improved with distortion: 0.004044861751026474
    > [11]: k-means improved with distortion: 0.004039863415528089
    > [21]: k-means improved with distortion: 0.004026258517114911
    > [91]: k-means improved with distortion: 0.0039976816333364695



```python
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap='rainbow', c=idx, s=8**2)
ax.set_title('Pixel dataset plotted in 3D.\nColor shows centroid memberships')

plt.show()
```


![png](out/output_33_0.png)



```python
X_norm, mu, sig = feature_normalization(X)
U, S = pca(X_norm)
Z = project_data(X_norm, U, K)

fig = plt.figure(figsize=(12, 10))

ax = fig.add_subplot(111)
ax.scatter(Z[:, 0], Z[:, 1], cmap='rainbow', c=idx, s=64)
ax.set_title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
ax.grid(False)

plt.show()
```


![png](out/output_34_0.png)


> На графиках по цветам кластеров видно соответствие между `3D` изображением и его `2D` проекции

### 19. conclusions

Был рассмотрен метод главных компонент. Произведены уменьшения размерности данных, сжатия картинок, показан результат работы метода главных компонент в сравнении с методом К-средних.
