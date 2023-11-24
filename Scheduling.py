import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Models import scheduling, DeepNeuralNetwork

r = np.array([11, 10, 8.2, 8, 13])
p = np.array([.1, .3, .25, .5, .7])
t = np.array([12, 7, 10, 5, 7.3])
T = 21
q = 0.67

obj = scheduling(r, p, t, T, q)

op, fp, er, TP = obj.optord()

plt.title("Expected Reward")
plt.plot(np.arange(0, 120), er, color="red")

plt.show()

plt.title("Probability of terminating before T")
plt.plot(np.arange(0, 120), TP, color="red")

plt.show()

print(fp)

N = 400

ov = np.zeros((N, 25))
input = np.zeros((N, 15))

for k in range(N):
    r = 10 * np.random.rand(5)
    p = np.random.rand(5)
    t = 10 * np.random.rand(5)
    obj = scheduling(r, p, t, T, q)
    op, fp, er, TP = obj.optord()
    ov[k, :] = op.flatten()
    input[k, :] = np.append(r, np.append(p, t))


model = DeepNeuralNetwork(15, 30, 28, 25)

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 90000
input=torch.from_numpy(input)
input = input.to(torch.float32)
ov = torch.from_numpy(ov)
ov = ov.to(torch.float32)

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input)

    loss = criterion(outputs, ov)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

bp = model.predict_binary(input)
