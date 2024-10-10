import torch
from torchvision.io import read_image
import torch.nn as nn
# **TIP: Reading an image file**




img = read_image('example-image.png')

print('Image shape:', img.shape)
print('Number of channels:', img.shape[0])
print('Image data type:', img.dtype)
print(img[:, 100:102, 100:102])

# Note that the img.dtype is uint8, which we can utilize into NumPy arrays to
# reduce memory usage compared to 16/32/64-bit integer types. Unsigned 8-bit
# takes values in the range [0, 255] which are sufficient to store pixel info
# in RGB images, which also takes value in the same range.

# Note that with torchvision, the input and output image tensors are in the format of
# Tensor[channels, image_height, image_width].


# ## Regularizing a neural network with L2 regularization and dropout
#
#
loss_func = nn.BCELoss()
loss = loss_func(torch.tensor([0.9]), torch.tensor([1.0]))
l2_lambda = 0.001

conv_layer = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5)
l2_penalty = l2_lambda * sum([(p**2).sum() for p in conv_layer.parameters()])
loss_with_penalty = loss + l2_penalty

linear_layer = nn.Linear(10, 16)
l2_penalty = l2_lambda * sum([(p**2).sum() for p in linear_layer.parameters()])
loss_with_penalty = loss + l2_penalty

# ## Loss Functions for Classification
#
#  * **`nn.BCELoss()`**
#    * `from_logits=False`
#    * `from_logits=True`
#
#  * **`nn.CrossEntropyLoss()`**
#    * `from_logits=False`
#    * `from_logits=True`
#


####### Binary Cross-entropy
# loss function for binary classification (with a single output unit)
logits = torch.tensor([0.8])
probas = torch.sigmoid(logits)
target = torch.tensor([1.0])

bce_loss_fn = nn.BCELoss()
bce_logits_loss_fn = nn.BCEWithLogitsLoss()

print(f'BCE (w Probas): {bce_loss_fn(probas, target):.4f}')
print(f'BCE (w Logits): {bce_logits_loss_fn(logits, target):.4f}')

####### Categorical Cross-entropy
# loss function for multiclass classification
# in torch.nn module the categorical cross-entropy takes in ground truth labels as integers
# (i.e., y=2 out of the three classes 0, 1, and 2)
logits = torch.tensor([[1.5, 0.8, 2.1]])
probas = torch.softmax(logits, dim=1)
target = torch.tensor([2])

cce_loss_fn = nn.NLLLoss()
cce_logits_loss_fn = nn.CrossEntropyLoss()

print(f'CCE (w Logits): {cce_logits_loss_fn(logits, target):.4f}')
print(f'CCE (w Probas): {cce_loss_fn(torch.log(probas), target):.4f}')

# Note that computing cross-entropy loss by providing the logits, and not the class-membership
# probabilities is usually preferred due to numerical stability reasons.
# For binary classification we can either provide logits as inputs to the loss function
# nn.BCEWithLogitsLoss(), or compute the probabilities based on the logits and feed them
# to the loss function nn.BCELoss().
# For multiclass classification, we either provide logits as inputs to the loss function nn.CrossEntropyLoss()
#, or compute the log probabilities based on the logits and feed them to the negative log-likelihood los function
# nn.NLLLoss().


