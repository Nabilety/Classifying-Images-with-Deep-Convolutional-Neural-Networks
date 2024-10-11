import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import DataLoader


# ## Smile classification from face images using CNN
#

# ### Loading the CelebA dataset

# You can try setting `download=True` in the code cell below, however due to the daily download limits of the CelebA dataset, this will probably result in an error. Alternatively, we recommend trying the following:
#
# - You can download the files from the official CelebA website manually (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
# - or use our download link, https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing (recommended).
#
# If you use our download link, it will download a `celeba.zip` file,
#
# 1. which you need to unpack in the current directory where you are running the code.
# 2. In addition, **please also make sure you unzip the `img_align_celeba.zip` file, which is inside the `celeba` folder.**
# 3. Also, after downloading and unzipping the celeba folder, you need to run with the setting `download=False` instead of `download=True` (as shown in the code cell below).
#
# In case you are encountering problems with this approach, please do not hesitate to open a new issue or start a discussion at https://github.com/ rasbt/machine-learning-book so that we can provide you with additional information.


image_path = './'
celeba_train_dataset = torchvision.datasets.CelebA(image_path, split='train', target_type='attr', download=True)
celeba_valid_dataset = torchvision.datasets.CelebA(image_path, split='valid', target_type='attr', download=True)
celeba_test_dataset = torchvision.datasets.CelebA(image_path, split='test', target_type='attr', download=True)

print('Train set:', len(celeba_train_dataset))
print('Validation set:', len(celeba_valid_dataset))
print('Test set:', len(celeba_test_dataset))


# ### Image transformation and data augmentation (data augmentation reduce overfitting and improve generalization performance)
# We will only be using a small portion of the training data (16000 training examples) to speed up the training process


## take 5 examples - with different types of transformation:
fig = plt.figure(figsize=(16, 8.5))
# 1) Column 1: cropping an image to a bounding box
ax = fig.add_subplot(2, 5, 1)
img, attr = celeba_train_dataset[0]
ax.set_title('Crop to a \nbounding-box', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 6)
img_cropped = transforms.functional.crop(img, 50, 20, 128, 128)
ax.imshow(img_cropped)

# 2) Column 2: flipping an image horizontally
ax = fig.add_subplot(2, 5, 2)
img, attr = celeba_train_dataset[1]
ax.set_title('Flip (horizontal)', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 7)
img_flipped = transforms.functional.hflip(img)
ax.imshow(img_flipped)

# 3) Column 3: adjusting the contrast
ax = fig.add_subplot(2, 5, 3)
img, attr = celeba_train_dataset[2]
ax.set_title('Adjust contrast', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 8)
img_adj_contrast = transforms.functional.adjust_contrast(img, contrast_factor=2)
ax.imshow(img_adj_contrast)

# 4) Column 4: adjusting the brightness
ax = fig.add_subplot(2, 5, 4)
img, attr = celeba_train_dataset[3]
ax.set_title('Adjust brightness', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 9)
img_adj_brightness = transforms.functional.adjust_brightness(img, brightness_factor=1.3)
ax.imshow(img_adj_brightness)

# 5) Column 5: center-croppping an image and resizing the resulting image back to its original size, (218, 178)
ax = fig.add_subplot(2, 5, 5)
img, attr = celeba_train_dataset[4]
ax.set_title('Center crop\nand resize', size=15)
ax.imshow(img)
ax = fig.add_subplot(2, 5, 10)
img_center_crop = transforms.functional.center_crop(img, [0.7 * 218, 0.7 * 178])
img_resized = transforms.functional.resize(img_center_crop, size=(218, 178))
ax.imshow(img_resized)

plt.show()


# In practice the data augmentation transformations are randomized during model training
# where we create a pipeline of these transformations. I.e. we can first randomly crop an image, flip it randomly, and
# finally resize it to the desired size:

torch.manual_seed(1)

fig = plt.figure(figsize=(14, 12))

for i, (img, attr) in enumerate(celeba_train_dataset):
    ax = fig.add_subplot(3, 4, i * 4 + 1)
    ax.imshow(img)
    if i == 0:
        ax.set_title('Orig.', size=15)

    ax = fig.add_subplot(3, 4, i * 4 + 2)
    img_transform = transforms.Compose([transforms.RandomCrop([178, 178])])
    img_cropped = img_transform(img)
    ax.imshow(img_cropped)
    if i == 0:
        ax.set_title('Step 1: Random crop', size=15)

    ax = fig.add_subplot(3, 4, i * 4 + 3)
    img_transform = transforms.Compose([transforms.RandomHorizontalFlip()])
    img_flip = img_transform(img_cropped)
    ax.imshow(img_flip)
    if i == 0:
        ax.set_title('Step 2: Random flip', size=15)

    ax = fig.add_subplot(3, 4, i * 4 + 4)
    img_resized = transforms.functional.resize(img_flip, size=(128, 128))
    ax.imshow(img_resized)
    if i == 0:
        ax.set_title('Step 3: Resize', size=15)

    if i == 2:
        break

plt.show()


# Note each iteration thorugh these three examples, we get slightly different images due to random transformations
# For convenience we can define transform functions to use this pipeline for data augmentation during dataset loading-

# Following code we define function get_smile, which will extract the smile label from the 'attributes' list

get_smile = lambda attr: attr[18]

# we will define the transform_train function that will produce the transformed image
# (where we wil first randomly crop the image, flip it randomly and finaly resize it to the desired 64x64 size)
transform_train = transforms.Compose([
    transforms.RandomCrop([178, 178]),
    transforms.RandomHorizontalFlip(),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

# Note we will only apply data augmentation to the training examples, however, and not to validation or test images
# The code for validation or teset set is as follows
# (where we first simply crop the image and then resize to desired 64x64 size)
transform = transforms.Compose([
    transforms.CenterCrop([178, 178]),
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
])

# Now we apply the transform_train function to our training dataset and iterate over the dataset five times:
celeba_train_dataset = torchvision.datasets.CelebA(image_path,
                                                   split='train',
                                                   target_type='attr',
                                                   download=False,
                                                   transform=transform_train,
                                                   target_transform=get_smile)

torch.manual_seed(1)
data_loader = DataLoader(celeba_train_dataset, batch_size=2)

fig = plt.figure(figsize=(15, 6))

num_epochs = 5
for j in range(num_epochs):
    img_batch, label_batch = next(iter(data_loader))
    img = img_batch[0]
    ax = fig.add_subplot(2, 5, j + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Epoch {j}:', size=15)
    ax.imshow(img.permute(1, 2, 0))

    img = img_batch[1]
    ax = fig.add_subplot(2, 5, j + 6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img.permute(1, 2, 0))
plt.show()

# This shows resulting transformations for data augmentation on two example images


# Next we will apply transform function to our validation and test datasets:
celeba_valid_dataset = torchvision.datasets.CelebA(image_path,
                                                   split='valid',
                                                   target_type='attr',
                                                   download=False,
                                                   transform=transform,
                                                   target_transform=get_smile)

celeba_test_dataset = torchvision.datasets.CelebA(image_path,
                                                   split='test',
                                                   target_type='attr',
                                                   download=False,
                                                   transform=transform,
                                                   target_transform=get_smile)

# instead of using all the available training and validation data, we will take a subset of
# 16000 training examples and 1000 validation examples, as our goal is to intentionally train our model with small dataset
celeba_test_dataset = Subset(celeba_test_dataset, torch.arange(16000))
celeba_valid_dataset = Subset(celeba_valid_dataset, torch.arange(1000))

print('Train set:', len(celeba_train_dataset))
print('Validation set:', len(celeba_valid_dataset))

# Create data loaders for three datasets
batch_size = 32

torch.manual_seed(1)
train_dl = DataLoader(celeba_train_dataset, batch_size, shuffle=True)
valid_dl = DataLoader(celeba_valid_dataset, batch_size, shuffle=False)
test_dl = DataLoader(celeba_test_dataset, batch_size, shuffle=False)
# Now that data loaders are ready, we will develop a CNN model and train and evaluate it next

# ### Training a CNN Smile classifier
#
# * **Global Average Pooling**
# The CNN model receives input images of size 3x64x64 (images have three color channels)
# The input data goes through four convolutional layers to make 32, 64, 128 and 256 feature maps using filters
# with a kernel size of 3x3 and padding of 1 for same padding. The first three convolution layers are followed
# by max-pooling, P_2x2. Two dropout layers are also included for regularization:
model = nn.Sequential()

model.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout1', nn.Dropout(p=0.5))

model.add_module('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
model.add_module('dropout2', nn.Dropout(p=0.5))

model.add_module('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
model.add_module('relu3', nn.ReLU())
model.add_module('pool3', nn.MaxPool2d(kernel_size=2))

model.add_module('conv4', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
model.add_module('relu4', nn.ReLU())

# Shape of output feature maps after applying these layers using a toy batch input (four arbitrary images)
x = torch.ones((4, 3, 64, 64))
print(model(x).shape)

# So 256 feature maps (or channels) of size 8x8.
# Now we can add a fully connected layer to get the output layer with a single unit
# If we reshape (flatten) the feature maps, the number of input units to this fully connected layer will be
# 8 x 8 x 256 = 16384

# Alternatively let's consider a new layer called global average-pooling, which compute the average of each
# feature map separately, thereby reducing the hidden units to 256.
# global average-pooling i conceptually very similar to other pooling layers. It's a special case of average-pooling
# when the pooling size is equal to the size of the input feature maps.

# Given our case of the shape of the feature maps prior to this layer is [batchsize x 256 x 8 x 8]
# we expect to get 256 units as output, that is teh shape of the output will be [batchsize x 256)
# We will add this layer and recomplute the output shape to verify if this is true:
model.add_module('pool4', nn.AvgPool2d(kernel_size=8))
model.add_module('flatten', nn.Flatten())

x = torch.ones((4, 3, 64, 64))
print(model(x).shape) # torch.Size([4, 256])

# Finally we can add the fully connected layer to get a single output unit
# In this case we specify the activation function to be sigmoid:
model.add_module('fc', nn.Linear(256, 1))
model.add_module('sigmoid', nn.Sigmoid())

x = torch.ones((4, 3, 64, 64))
print(model(x).shape) # torch.Size([4, 1])

print(model)

device = torch.device("cuda:0")
# device = torch.device("cpu")
model = model.to(device)

# Next step we create a loss function and optimizer (Adam optimizer again)
# For binary classification with single probabilistic output, we use BCELoss as loss function:
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Now we can train the model
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
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu()

        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = ((pred >= 0.5).float() == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu()

        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)

        print(
            f'Epoch {epoch + 1} accuracy: {accuracy_hist_train[epoch]:.4f} val_accuracy: {accuracy_hist_valid[epoch]:.4f}')
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


# We train the CNN mode lwith 30 epochs and use validation dataset that we created for monitoring the learning progress
torch.manual_seed(1)
num_epochs = 30
hist = train(model, num_epochs, train_dl, valid_dl)

# Visualize learning curve and compare the training and validation loss and accuracies after each epoch
x_arr = np.arange(len(hist[0])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist[0], '-o', label='Train loss')
ax.plot(x_arr, hist[1], '--<', label='Validation loss')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(x_arr, hist[2], '-o', label='Train acc.')
ax.plot(x_arr, hist[3], '--<', label='Validation acc.')
ax.legend(fontsize=15)
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Accuracy', size=15)

plt.show()


# Once we are happy with the learning curve, we can evaluate the model on the hold-out test dataset
accuracy_test = 0

model.eval()
with torch.no_grad():
    for x_batch, y_batch in test_dl:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)[:, 0]
        is_correct = ((pred >= 0.5).float() == y_batch).float()
        accuracy_test += is_correct.sum().cpu()

accuracy_test /= len(test_dl.dataset)

print(f'Test accuracy: {accuracy_test:.4f}')

# Finally we will take a small subset of 10 examples from the last batch of our pre-processed test dataset (test_dl)
# and compute the probabilities of each example of being from class 1 (corresponding to smile) and visualize the
# examples along with their ground truth label and predicted probabilities
pred = model(x_batch)[:, 0] * 100

fig = plt.figure(figsize=(15, 7))
for j in range(10, 20):
    ax = fig.add_subplot(2, 5, j - 10 + 1)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(x_batch[j].cpu().permute(1, 2, 0))
    if y_batch[j] == 1:
        label = 'Smile'
    else:
        label = 'Not Smile'
    ax.text(
        0.5, -0.15,
        f'GT: {label:s}\nPr(Smile)={pred[j]:.0f}%',
        size=16,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes)

plt.show()

# As seen the 10 example images along with their ground truth labels and probabilities that they belong to class 1, smile
# If replaced the global average-pooling with a fully connected layer, and used the entire trainign dataset with CNN
# architecture we trained, we should be able to achieve above 90% accuracy