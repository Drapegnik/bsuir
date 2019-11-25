#!/usr/bin/env python
# coding: utf-8

# # ml lab9

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io


# ### 1. read `ex9_movies.mat` data

# In[4]:


data = scipy.io.loadmat('data/ex9_movies.mat')
Y = data['Y']
R = data['R']

num_movies, num_users = Y.shape
num_movies, num_users


# ### 2. number of featurs

# In[7]:


NUM_FEATURES = 10


# ### 3-6. cost function + gradient

# In[9]:


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


# ### 7. train model with `scipy`

# In[13]:


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


# In[14]:


train(Y, R, lambda_=10)


# ### 8. add own ratings

# In[26]:


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


# In[41]:


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


# ### 9. get recommendations

# In[42]:


def normalize_ratings(Y, R):
    n = Y.shape[0]
    Ymean = np.zeros(n)
    Ynorm = np.zeros(Y.shape)

    for i in range(n):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean


# In[47]:


Y = np.hstack([my_ratings_list[:, None], Y])
R = np.hstack([(my_ratings_list > 0)[:, None], R])
Ynorm, Ymean = normalize_ratings(Y, R)

X, theta = train(Ynorm, R, lambda_=10)
p = np.dot(X, theta.T)


# In[50]:


def print_predict(p, n=30):
    predict = p[:, 0] + Ymean
    idx = np.argsort(predict)[::-1]

    for i in range(n):
        print(f'{i+1}.\t{predict[idx[i]]:.2f}\t{MOVIES_NAMES[idx[i]]}')
        
print_predict(p)


# > Сложно оценить правильность рекомендаций так как я не смотрел много фильмов до 2000 года выпуска

# ### 10. train with singular vectors

# In[52]:


from scipy.sparse.linalg import svds

U, sigma, Vt = svds(Y, NUM_FEATURES)
sigma = np.diag(sigma)
p = U.dot(sigma).dot(Vt)


# In[53]:


print_predict(p)


# > Результаты незначительно отличаются

# ### 11. conclusions

# Была рассмотрена и реализована рекомендательная система с использованием алгоритма коллаборативной фильтрации.
# 
# Получены рекомендации на основе собственных оценок.
# 
# Также получены рекомендации с помощью сингулярного разложения матриц.

# In[ ]:




