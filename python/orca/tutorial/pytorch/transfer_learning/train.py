#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Most of the pytorch code is adapted from PyTorch's transfer learning tutorial for
# hymenoptera dataset.
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

plt.ion()  # interactive mode

# Step 1: Init Orca Context

from bigdl.orca import init_orca_context, stop_orca_context
init_orca_context()

# Step 2: Define Train Dataset

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=1)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cpu")


def train_loader_func(config, batch_size):
    return dataloaders['train']


def test_loader_func(config, batch_size):
    return dataloaders['val']


def imshow(inp, title=None):
    """
    Visualize a few training images 
    so as to understand the data augmentations.
    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

inputs, classes = next(iter(dataloaders['train']))  # Get a batch of training data
out = torchvision.utils.make_grid(inputs)  # Make a grid from batch
imshow(out, title=[class_names[x] for x in classes])

# Step 3: Finetuning the Convnet

# Instead of random initialization, we initialize the network with a pretrained network,
# like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual.

# Step 3.1: Define the Model

from model import ConvNetModel

# Step 3.2: Finetune with Orca Estimator

from bigdl.orca.learn.pytorch import Estimator 
from bigdl.orca.learn.metrics import Accuracy

# Create the estimator
est = Estimator.from_torch(model=ConvNetModel.model_creator,
                           optimizer=ConvNetModel.optimizer_creator,
                           loss=nn.CrossEntropyLoss(),
                           metrics=[Accuracy()],
                           scheduler_creator=ConvNetModel.scheduler_creator,
                           scheduler_step_freq="epoch",
                           config={'device': device},
                           use_tqdm=True, backend="ray")

# Fit the estimator
est.fit(data=train_loader_func, epochs=5, validation_data=test_loader_func)

# Step 3.3: Save and Evaluate the Model

# Save the model
est.save("finetuned_convnet_model")

# Evaluate the model
result = est.evaluate(data=test_loader_func)
for r in result:
    print(r, ":", result[r])


def visualize_model(model, num_images=6):
    """
    Visualize the model predictions
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_model(est.get_model())

# Shut down workers and releases resources
est.shutdown()

# Step 4: ConvNet as fixed feature extractor

# Here, we will freeze the weights for all of the network except that of
# the final fully connected layer. This last fully connected layer is
# replaced with a new one with random weights and only this layer is trained.

# Step 4.1: Define the Model

from model import FixedConvNetModel

# Step 4.2: Finetune with Orca Estimator

est_fixed = Estimator.from_torch(model=FixedConvNetModel.model_creator,
                                 optimizer=FixedConvNetModel.optimizer_creator,
                                 loss=nn.CrossEntropyLoss(), metrics=[Accuracy()],
                                 scheduler_creator=FixedConvNetModel.scheduler_creator,
                                 scheduler_step_freq="epoch", config={'device': device},
                                 use_tqdm=True, backend="ray")

est_fixed.fit(data=train_loader_func, epochs=5, validation_data=test_loader_func)

# Step 4.3: Save and Evaluate the Model

est_fixed.save("finetuned_fixed_convnet_model")

result = est_fixed.evaluate(data=test_loader_func)
for r in result:
    print(r, ":", result[r])

visualize_model(est_fixed.get_model())

est_fixed.shutdown()

# Step 5: Stop orca context

stop_orca_context()
