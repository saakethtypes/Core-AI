import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets

# Statics 
epochs = 1000
lr = 0.01

# Preprocess data
X,Y = datasets.make_regression(n_samples = 100,n_features = 1, noise = 20, random_state= 1)
X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))
Y = Y.view(Y.shape[0],1)
n_samples,n_features = X.shape

# Model
input_size = n_features
output_size = 1 
model = nn.Linear(input_size,output_size)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training Loop
for epoch in range(epochs):
    #forward pass
    y_pred = model(X)
    loss = criterion(y_pred,Y)
    #back pass
    loss.backward() 
    #update weights
    optimizer.step()
    #clear grad
    optimizer.zero_grad()
    #training info
    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Visualize Predictions
with torch.no_grad():
    predicted = model(X).detach().numpy()
    plt.plot(X,Y,'ro')
    plt.plot(X,y_pred,'b')
    plt.show()