import numpy as np

N, p = 30, 20
np.random.seed(0)
X = np.random.randn(N, p)
Y = 2 * np.random.randint(2, size = N) - 1 

theta = np.zeros(p)
lambd  = 0.1
epochs = 50 
learning_rate = 0.01    

for epoch in range(epochs):
    for i in range(N):
        xi = X[i]
        yi = Y[i]
        
        margin = yi * np.dot(xi, theta)
        
        if(margin >=1) :
            gradient = 2 * lambd * theta
        else:
            gradient = -yi * xi + 2 * lambd * theta
            
        theta -= learning_rate * gradient
        
print("Final learned theta:", theta)        
print("Final loss:", np.mean(np.log(1 + np.exp(-Y * (np.dot(X, theta))))))
print("Final accuracy:", np.mean((np.dot(X, theta) > 0) == Y))
print("Final predictions:", ((np.dot(X, theta))> 0).astype(int))
