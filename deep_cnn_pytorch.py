import torch
import numpy as np
from torchvision.io import read_image
import torch.nn as nn
import torchvision
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
# ## Implementing a deep convolutional neural network using PyTorch
#
# ### The multilayer CNN architecture





# ### Loading and preprocessing the data (MNIST)
image_path = './'
transform = transforms.Compose([transforms.ToTensor()])

mnist_dataset = torchvision.datasets.MNIST(root=image_path,
                                           train=True,
                                           transform=transform,
                                           download=False)

mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000)) # first 10k as examples as validation
mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
mnist_test_dataset = torchvision.datasets.MNIST(root=image_path,
                                           train=False,
                                           transform=transform,
                                           download=False)

# Construct data loader with batches of 64 images for training set and validation set
batch_size = 64
torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)
# the features we read are of values in the range [0, 1]
# We also already converted images to tensors
# the labels are integers from 0 to 9 representing ten digits, hence we don't need any scaling/conversions

# ### Implementing a CNN using the torch.nn module
#
# #### Configuring CNN layers in PyTorch
#
#  * **Conv2d:** `torch.nn.Conv2d`
#    * `out_channels`
#    * `kernel_size`
#    * `stride`
#    * `padding`
#
#
#  * **MaxPool2d:** `torch.nn.MaxPool2d`
#    * `kernel_size`
#    * `stride`
#    * `padding`
#
#
#  * **Dropout** `torch.nn.Dropout`
#    * `p`

# ### Constructing a CNN in PyTorch

model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))

# calculate size of feature maps using PyTorch:
x = torch.ones((4, 1, 28, 28))
print(model(x).shape)

model.add_module('flatten', nn.Flatten())

x = torch.ones((4, 1, 28, 28))
print(model(x).shape)

# Add fully connected layer for implementing a classifier on top of our convolutional and pooling layers
# the input to this layer must have rank 2, that is, shape [batch_size x input_units]
# so we need to flatten the output of the previous layers to meet this requirement for the fully connected layer
model.add_module('fc1', nn.Linear(3136, 1024))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(p=0.5))

# Last fully connected layer has 10 output units for the 10 class labels in the MNIST dataset
# in practice we usually use softmax activation to obtain class-membership probabilities of each input example,
# assuming that the classes are mutually exclusive, so the probabilities for each example sum to 1
model.add_module('fc2', nn.Linear(1024, 10))
# However the softmax function is already used internally inside PyTorch's CrossEntropyLoss implementation
# which is why we don't have to explicitly add it as a layer after the output layer above

device = torch.device("cuda:0")
device = torch.device("cpu")

model = model.to(device)

# Create loss function and optimizer for the model
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# key advantage of Adam is in the choice of update step size derived from the running average of gradient moments


# Now we can train the model by defining following function:
def train(model, num_epochs, train_dl, valid_dl):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(
            f'Epoch {epoch + 1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


torch.manual_seed(1)
num_epochs = 20
hist = train(model, num_epochs, train_dl, valid_dl)

# Note that using the designated settings for training model.train() and evaluation model.eval() will
# automatically set the mode for the dropout layer and rescale the hidden units appropriately so we
# won't have to worry about them at all.

# Train CNN model and use the validation dataset we created for monitoring the learning progress.
print(hist[0])
print("Hist above")
# Visualize the learning curves
x_arr = np.arange(len(hist[0])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.legend(fontsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.show()

# evaluate trained model on the test datast:
torch.cuda.synchronize()
model_cpu = model.cpu()
pred = model(mnist_test_dataset.data.unsqueeze(1) / 255.)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f'Test accuracy: {is_correct.mean():.4f}')
# The CNN model achieves an accuracy of 99.07%. We got approximately 95% accuracy using only fully connected layers

# Get the prediction results in the form of class-membership probabilities and conver to predicted labels
# using torch.argmax function to find the elemnt with the maximum probability. Do this for batch of 12 examples
# and visualize the input and predicted labels
fig = plt.figure(figsize=(12, 4))
for i in range(12):
    ax = fig.add_subplot(2, 6, i + 1)
    ax.set_xticks([]);
    ax.set_yticks([])
    img = mnist_test_dataset[i][0][0, :, :]
    pred = model(img.unsqueeze(0).unsqueeze(1))
    y_pred = torch.argmax(pred)
    ax.imshow(img, cmap='gray_r')
    ax.text(0.9, 0.1, y_pred.item(),
            size=15, color='blue',
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)

plt.show()
