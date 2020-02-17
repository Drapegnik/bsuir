# ml lab9

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
```

### 1. read `ex9_movies.mat` data


```python
data = scipy.io.loadmat('data/ex9_movies.mat')
Y = data['Y']
R = data['R']

num_movies, num_users = Y.shape
num_movies, num_users
```




    (1682, 943)



### 2. number of featurs


```python
NUM_FEATURES = 10
```

### 3-6. cost function + gradient


```python
def get_reg_term(X, theta, lambda_):
    return (lambda_ / 2) * np.sum(np.square(X)) + (lambda_ / 2) * np.sum(np.square(theta))

def calc_cost(X, Y, R, theta):
    return (1 / 2) * np.sum(np.square((X.dot(theta.T) - Y) * R))


def cost_function(params, Y, R, num_users, num_movies, num_features, lambda_):
    X = params[:num_movies * num_features].reshape(num_movies, num_features)
    theta = params[num_movies * num_features:].reshape(num_users, num_features)

    X_grad = np.zeros(X.shape)
    theta_grad = np.zeros(theta.shape)

    J = calc_cost(X, Y, R, theta) + get_reg_term(X, theta, lambda_)

    for i in range(num_movies):
        idx = np.where(R[i, :] == 1)[0]
        theta_i = theta[idx, :]
        Y_i = Y[i, idx]
        X_grad[i, :] = (X[i, :].dot(theta_i.T) - Y_i).dot(theta_i) + lambda_ * X[i, :]

    for j in range(num_users):
        idx = np.where(R[:, j] == 1)[0]
        X_j = X[idx, :]
        Y_j = Y[idx, j]
        theta_grad[j, :] = (X_j.dot(theta[j, :]) - Y_j).dot(X_j) + lambda_ * theta[j, :]

    grad = np.concatenate([X_grad.ravel(), theta_grad.ravel()])
    return J, grad
```

### 7. train model with `scipy`


```python
import scipy.optimize

def train(Y, R, num_features = NUM_FEATURES, lambda_=0.0):
    num_movies, num_users = Y.shape
    initial_X = np.random.randn(num_movies, num_features)
    initial_theta = np.random.randn(num_users, num_features)

    cost_f = lambda x: cost_function(x, Y, R, num_users, num_movies, num_features, lambda_)
    initial_params = np.concatenate([initial_X.ravel(), initial_theta.ravel()])

    # Truncated Newton Algorithm
    params = scipy.optimize.minimize(cost_f, initial_params, method='TNC', jac=True).x

    idx = num_movies * num_features
    theta = params[idx:].reshape(num_users, num_features)
    X = params[:idx].reshape(num_movies, num_features)

    return X, theta
```


```python
train(Y, R, lambda_=10)
```




    (array([[ 1.4438716 , -0.39240415, -0.43620424, ...,  0.52014848,
              0.06694878, -0.16276768],
            [ 0.84902957, -0.30851568, -0.27051436, ...,  0.23291525,
              0.25925657,  0.20671643],
            [ 0.63175791, -0.00927988,  0.52589506, ..., -0.47434056,
              0.11256896, -0.05332482],
            ...,
            [ 0.12733217, -0.04512815,  0.0130063 , ..., -0.11735913,
             -0.06047865, -0.11521279],
            [ 0.13715983, -0.18520116,  0.00823545, ..., -0.05045468,
              0.07466929,  0.04718126],
            [ 0.25144985, -0.13913787,  0.03557105, ..., -0.03601842,
             -0.014978  ,  0.02562142]]),
     array([[ 1.52693186, -0.75135164,  0.40942174, ...,  0.00504388,
              0.40482936, -0.13868873],
            [ 1.07152901, -0.35972028, -0.49640467, ..., -0.0479072 ,
             -0.06016762,  0.18030974],
            [ 0.88823404, -0.52252648, -0.34845443, ...,  0.32460284,
              0.28596222, -0.31952627],
            ...,
            [ 1.23947958, -0.36239146, -0.16369162, ...,  0.03706565,
              0.10257671, -0.19273455],
            [ 1.10837565, -0.27744029, -0.41411452, ...,  0.05353096,
              0.21606041,  0.76446523],
            [ 0.578459  , -0.78164048, -0.27596438, ..., -0.34310137,
             -0.17185575, -0.56158548]]))



### 8. add own ratings


```python
MOVIES_IDS = {}
MOVIES_NAMES = {}

with open('data/movie_ids.txt',  encoding='ISO-8859-1') as file:
    movies = file.readlines()

    for movie in movies:
        _id, _name = movie.split(' ', 1)
        id = int(_id)
        name = _name.strip()
        MOVIES_IDS[name] = id
        MOVIES_NAMES[id] = name

MOVIES_IDS['Toy Story (1995)']
```




    1




```python
MY_RATINGS = {
    'Toy Story (1995)': 5,
    'Godfather, The (1972)': 2,
    'Home Alone (1990)': 4,
    'Pulp Fiction (1994)': 5,
    'Star Wars (1977)': 5,
    'Titanic (1997)': 2,
    'Men in Black (1997)': 5,
    'Turbo: A Power Rangers Movie (1997)': 1,
    '101 Dalmatians (1996)': 3,
    'Indiana Jones and the Last Crusade (1989)': 3,
    'Back to the Future (1985)': 5,
    'Wallace & Gromit: The Best of Aardman Animation (1996)': 3,
    'Forrest Gump (1994)': 4,
    'Taxi Driver (1976)': 3,
}

my_ratings_list = np.zeros(num_movies)

for name, rate in MY_RATINGS.items():
    id = MOVIES_IDS[name]
    my_ratings_list[id] = rate

my_ratings_list
```




    array([0., 5., 0., ..., 0., 0., 0.])



### 9. get recommendations


```python
def normalize_ratings(Y, R):
    n = Y.shape[0]
    Ymean = np.zeros(n)
    Ynorm = np.zeros(Y.shape)

    for i in range(n):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean
```


```python
Y = np.hstack([my_ratings_list[:, None], Y])
R = np.hstack([(my_ratings_list > 0)[:, None], R])
Ynorm, Ymean = normalize_ratings(Y, R)

X, theta = train(Ynorm, R, lambda_=10)
p = np.dot(X, theta.T)
```


```python
def print_predict(p, n=30):
    predict = p[:, 0] + Ymean
    idx = np.argsort(predict)[::-1]

    for i in range(n):
        print(f'{i+1}.\t{predict[idx[i]]:.2f}\t{MOVIES_NAMES[idx[i]]}')

print_predict(p)
```

    1.	5.00	Celluloid Closet, The (1995)
    2.	5.00	City of Industry (1997)
    3.	5.00	Temptress Moon (Feng Yue) (1996)
    4.	5.00	Simple Wish, A (1997)
    5.	5.00	Kim (1950)
    6.	5.00	Young Guns II (1990)
    7.	5.00	Enfer, L' (1994)
    8.	5.00	Umbrellas of Cherbourg, The (Parapluies de Cherbourg, Les) (1964)
    9.	5.00	Margaret's Museum (1995)
    10.	5.00	Grosse Fatigue (1994)
    11.	4.63	My Favorite Season (1993)
    12.	4.58	Midnight in the Garden of Good and Evil (1997)
    13.	4.58	In the Name of the Father (1993)
    14.	4.52	Santa Clause, The (1994)
    15.	4.51	Death in Brunswick (1991)
    16.	4.50	Of Human Bondage (1934)
    17.	4.50	Dadetown (1995)
    18.	4.49	Some Like It Hot (1959)
    19.	4.47	Twister (1996)
    20.	4.46	Monty Python and the Holy Grail (1974)
    21.	4.45	Horseman on the Roof, The (Hussard sur le toit, Le) (1995)
    22.	4.44	Spy Hard (1996)
    23.	4.41	American in Paris, An (1951)
    24.	4.37	Ruby in Paradise (1993)
    25.	4.37	Dances with Wolves (1990)
    26.	4.37	Around the World in 80 Days (1956)
    27.	4.36	Transformers: The Movie, The (1986)
    28.	4.35	Princess Bride, The (1987)
    29.	4.35	Muppet Treasure Island (1996)
    30.	4.35	Philadelphia Story, The (1940)


> Сложно оценить правильность рекомендаций так как я не смотрел много фильмов до 2000 года выпуска

### 10. train with singular vectors


```python
from scipy.sparse.linalg import svds

U, sigma, Vt = svds(Y, NUM_FEATURES)
sigma = np.diag(sigma)
p = U.dot(sigma).dot(Vt)
```


```python
print_predict(p)
```

    1.	5.01	Young Guns II (1990)
    2.	5.01	Celluloid Closet, The (1995)
    3.	5.00	Grosse Fatigue (1994)
    4.	5.00	Simple Wish, A (1997)
    5.	5.00	Kim (1950)
    6.	5.00	Temptress Moon (Feng Yue) (1996)
    7.	5.00	City of Industry (1997)
    8.	5.00	Umbrellas of Cherbourg, The (Parapluies de Cherbourg, Les) (1964)
    9.	5.00	Margaret's Museum (1995)
    10.	4.99	Enfer, L' (1994)
    11.	4.67	I.Q. (1994)
    12.	4.63	My Favorite Season (1993)
    13.	4.59	Spy Hard (1996)
    14.	4.58	Monty Python and the Holy Grail (1974)
    15.	4.56	Midnight in the Garden of Good and Evil (1997)
    16.	4.53	In the Name of the Father (1993)
    17.	4.52	Horseman on the Roof, The (Hussard sur le toit, Le) (1995)
    18.	4.52	Princess Bride, The (1987)
    19.	4.51	Twister (1996)
    20.	4.51	Dadetown (1995)
    21.	4.50	Of Human Bondage (1934)
    22.	4.50	Death in Brunswick (1991)
    23.	4.48	Delicatessen (1991)
    24.	4.48	Santa Clause, The (1994)
    25.	4.47	Some Like It Hot (1959)
    26.	4.46	Seven (Se7en) (1995)
    27.	4.45	Sleepless in Seattle (1993)
    28.	4.42	Starship Troopers (1997)
    29.	4.42	Empire Strikes Back, The (1980)
    30.	4.42	Annie Hall (1977)


> Результаты незначительно отличаются

### 11. conclusions

Была рассмотрена и реализована рекомендательная система с использованием алгоритма коллаборативной фильтрации.

Получены рекомендации на основе собственных оценок.

Также получены рекомендации с помощью сингулярного разложения матриц.
