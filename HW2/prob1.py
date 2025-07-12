import numpy as np

N, p = 30, 20
np.random.seed(0)
X = np.random.randn(N, p)
Y = 2 * np.random.randint(2, size = N)

theta = np.zeros(p)

epochs = 100 
learning_rate = 0.1    

for epoch in range(epochs):
    for i in range(N):
        xi = X[i]
        yi = Y[i]
        z = xi * np.dot(theta, xi)
        gradient = - yi*xi/(1+np.exp(z))
        theta -= learning_rate * gradient                               
        
print("Final theta:", theta)        