import numpy as np
# import matplotlib.pyplot as plt
# import sklearn.decomposition as PCA

N, p = 30, 20
np.random.seed(0)
X = np.random.randn(N, p)
Y = 2 * np.random.randint(2, size = N) - 1

theta = np.zeros(p)

epochs = 50 
learning_rate = 0.01    

for epoch in range(epochs):
    for i in range(N):
        xi = X[i]
        yi = Y[i]
        z = xi * np.dot(theta, xi)
        gradient = - yi*xi/(1+np.exp(z))
        theta -= learning_rate * gradient                               
        
print("Final learned theta:", theta)        
print("Final loss:", np.mean(np.log(1 + np.exp(-Y * (np.dot(X, theta))))))
print("Final accuracy:", np.mean((np.dot(X, theta) > 0) == Y))
print("Final predictions:", ((np.dot(X, theta))> 0).astype(int))

# loss_history = []

# # Logistic loss function
# def logistic_loss(X, Y, theta):
#     z = Y * (np.dot(X, theta))
#     return np.mean(np.log(1 + np.exp(-z)))

# # SGD training
# for epoch in range(epochs):
#     for i in range(N):
#         xi = X[i]
#         yi = Y[i]
#         z = yi * np.dot(xi, theta)
#         grad = -yi * xi / (1 + np.exp(z))
#         theta -= learning_rate * grad
#     # Record loss after each epoch
#     loss_history.append(logistic_loss(X, Y, theta))

# # ---- Plot loss curve ----
# plt.figure(figsize=(7, 4))
# plt.plot(range(1, epochs + 1), loss_history, marker='o')
# plt.title("Logistic Loss over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Average Logistic Loss")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ---- Visualize Decision Boundary (PCA to 2D) ----
# # Reduce X and theta to 2D
# pca = PCA(n_components=2)
# X_2D = pca.fit_transform(X)
# # theta_2D = pca.components_ @ theta  # project theta to 2D
# theta_2D = np.dot(pca.components_.T, theta)  # ensure theta is in the same space as X_2D

# # Plot
# plt.figure(figsize=(6, 6))
# for label in [-1, 1]:
#     plt.scatter(X_2D[Y == label, 0], X_2D[Y == label, 1], label="Class {label}".format, alpha=0.7)

# w0, w1 = theta_2D
# x_vals = np.linspace(X_2D[:, 0].min(), X_2D[:, 0].max(), 100)
# y_vals = -(w0 / w1) * x_vals
# plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')

# plt.legend()
# plt.title("Decision Boundary (after PCA to 2D)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()