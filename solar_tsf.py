import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

data = np.genfromtxt('solar_AL.csv', delimiter=',', dtype=float)
data_small = [row[0] for row in data[25000: 35000]]
data_tiny = data_small[0 : 1008]
target = data_small[1008 : 2016]
data_tiny_test = data_small[2016 : 3024]
test_target = data_small[3024 : 4032]


data_tiny = torch.tensor(np.array(data_tiny).reshape(1, 1008),  dtype=torch.float32)
target = torch.tensor(np.array(target).reshape(1, 1008),  dtype=torch.float32)

data_tiny_test = torch.tensor(np.array(data_tiny_test).reshape(1, 1008),  dtype=torch.float32)
test_target = torch.tensor(np.array(test_target).reshape(1, 1008),  dtype=torch.float32)

# Create a linear layer with 10 input features and 5 output features.
linear_layer = nn.Linear(1008, 1008)

# Create a loss function.
loss_function = nn.MSELoss()

# Create an optimizer.
optimizer = torch.optim.SGD(linear_layer.parameters(), lr=0.00000001)

loss_tracker = []
test_loss_tracker = []
# Train the linear layer for 100 epochs.
for epoch in range(1000):

    # Calculate the output of the linear layer.
    output_tensor = linear_layer(data_tiny)
    test_output_tensor = linear_layer(data_tiny_test)

    # Calculate the loss.
    loss = loss_function(output_tensor, target)
    test_loss = loss_function(test_output_tensor, test_target)

    # Backpropagate the loss.
    loss.backward()

    # Update the weights of the linear layer.
    optimizer.step()

    # Print the loss after training.
    print(loss.item())
    loss_tracker.append(loss.item())
    test_loss_tracker.append(test_loss.item())
    
    if loss.item() < 0.001:
        break


linear_layer(data_tiny)

# print(target.detach().numpy().reshape(50))
# print(linear_layer(data_tiny).detach().numpy().reshape(50))

# plt.plot(loss_tracker)
# plt.plot([target.detach().numpy().reshape(50),linear_layer(data_tiny).detach().numpy().reshape(50)])
# plt.plot(linear_layer(data_tiny).detach().numpy().reshape(50))
# plt.plot(target.detach().numpy().reshape(50))


# Plot the results.
# plt.plot(target.detach().numpy().reshape(1008), label='target')
# plt.plot(linear_layer(data_tiny).detach().numpy().reshape(1008), label='linear_layer(data_tiny)')
# plt.plot(test_target.detach().numpy().reshape(1008), label='test_target')
# plt.plot(linear_layer(data_tiny_test).detach().numpy().reshape(1008), label='linear_layer(data_tiny_test)')


plt.plot(loss_tracker, label='loss_tracker')
plt.plot(test_loss_tracker, label='test_loss_tracker')
plt.legend()
plt.show()