#!/usr/bin/env python
# coding: utf-8

# # ml lab1

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### 1. read data

# In[104]:


data1 = np.genfromtxt('data/ex1data1.txt', delimiter=',')
rest = pd.DataFrame(data1, columns=['Population', 'Income'])
print(rest)


# ### 2. plot data

# In[264]:


def get_plot():
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.ylabel('Income, $10.000')
    plt.xlabel('Population of City, 10.000')
    plt.plot(
        rest['Population'],
        rest['Income'],
        'rx', markersize=10, label='Training Data')
    return plt


plot = get_plot()
plot.show()


# ### 3. cost function J(θ)

# In[289]:


# Linear hypothesis function
def h(X, theta):
    return np.dot(X, theta)


# J = compute_cost(X, y, theta)
# computes the cost of using theta as the parameter for linear regression
# to fit the data points in X and y
def compute_cost(X, y, theta):
    m = y.size
    return np.sum(np.square(h(X, theta) - y)) / (2. * m)


# In[290]:


(_, n) = rest.shape
theta = np.zeros((1, n)).T

X1 = rest[['Population']]
X1.insert(0, 'theta_0', 1)
y1 = rest[['Income']]

J = compute_cost(X1, y1, theta)
print(f'theta:\t{theta.ravel()}\nJ:\t{float(J)}')


# ### 4. gradient descent

# In[291]:


# Performs gradient descent to learn theta
def gradient_descent(X, y, theta, alpha=0.01, iterations=1500):
    m = y.size
    J_history = []

    for i in range(0, iterations):
        error = h(X, theta) - y
        theta -= alpha * np.dot(X.T, error) / m
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history


theta, j_history = gradient_descent(X1, y1, theta)
print(f'computed theta: {theta.ravel()}')


# In[246]:


sample_population = [3, 11, 15, 16, 18.5]
predicted_income = [np.dot([1, x], theta).sum() for x in sample_population]
pd.DataFrame(
    zip(sample_population, predicted_income),
    columns=['Sample Population', 'Predicted Income'])


# In[277]:


h_values = [np.dot(x, theta).sum() for x in X1.to_numpy()]

plot = get_plot()
plot.plot(rest['Population'], h_values, 'b-', label='Hypothesis')
plot.legend()
plot.show()


# ###  5. visualizing J(θ)

# In[284]:


# grid coordinates for plotting
xvals = np.linspace(-10, 10, 50)
yvals = np.linspace(-1, 4, 50)
xx, yy = np.meshgrid(xvals, yvals, indexing='xy')
Z = np.zeros((xvals.size, yvals.size))

# calculate Z-values (Cost) based on grid of coefficients
for (i, j), v in np.ndenumerate(Z):
    Z[i, j] = compute_cost(X1, y1, theta=[[xx[i, j]], [yy[i, j]]])


# In[294]:


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
