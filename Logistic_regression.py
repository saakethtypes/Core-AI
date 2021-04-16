import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Statics
epochs = 100
lr = 0.01

# Preprocess Data
# load data
bc = datasets.load_breast_cancer()
X_bc,Y_bc = bc.data,bc.target
# test train split
n_samples, n_features = X_bc.shape
X,x,Y,y = train_test_split(X_bc,Y_bc,test_size=0.2,random_state=1234)
# normalize
scaling = StandardScaler()
X = scaling.fit_transform(X)
x = scaling.transform(x)
# convert to floats
X = torch.from_numpy(X.astype(np.float32))
x = torch.from_numpy(x.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32))
# reshape y
Y = Y.view(Y.shape[0],1)
y = y.view(y.shape[0],1)

# Model 
class Logistic(nn.Module):
    def __init__(self,n_input_features):
        super(Logistic,self).__init__()
        self.linear = nn.Linear(n_input_features,1)
    def forward(self,x):
        layer_1_activations = self.linear(x)
        y_pred = torch.sigmoid(layer_1_activations)
        return y_pred

model = Logistic(n_features)

# Loss and Optimizer 
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=lr)

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
    model.zero_grad()
    #trianing info
    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
# Visualize Predictions
with torch.no_grad():
    predicted = model(x)
    y_preds_cls = predicted.round()
    acc = y_preds_cls.eq(y).sum() / float(y.shape[0])
    print(f'accuracy: {acc:.4f}')
