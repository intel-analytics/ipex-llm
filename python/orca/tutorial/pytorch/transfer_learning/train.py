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

# Step 0: Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.orca.learn.pytorch import Estimator
from bigdl.orca.learn.metrics import Accuracy

plt.ion()  # interactive mode


# Step 1: Init Orca Context
sc = init_orca_context()


# Step 2: Define train and test datasets as PyTorch DataLoader
def load_dataset(dataset_dir, batch_size=4):
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=1)
                   for x in ['train', 'val']}
    train_loader = dataloaders['train']
    test_loader = dataloaders['val']
    return train_loader, test_loader, class_names


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


def train_loader_func(config, batch_size):
    train_loader, test_loader, class_names = load_dataset(config['data_dir'], batch_size)

    inputs, classes = next(iter(train_loader))  # Get a batch of training data
    out = torchvision.utils.make_grid(inputs)  # Make a grid from batch
    imshow(out, title=[class_names[x] for x in classes])
    return train_loader


def test_loader_func(config, batch_size):
    train_loader, test_loader, class_names = load_dataset(config['data_dir'], batch_size)
    return test_loader


# Step 3: Finetuning the Convnet

# Instead of random initialization, we initialize the network with a pretrained network,
# like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual.


# Step 3.1: Define the model, optimizer and loss
from model import ConvNetModel


# Step 3.2: Finetune with Orca PyTorch Estimator
data_dir = 'hymenoptera_data'
backend = "ray"
device = torch.device("cpu")

est = Estimator.from_torch(model=ConvNetModel.model_creator,
                           optimizer=ConvNetModel.optimizer_creator,
                           loss=nn.CrossEntropyLoss(),
                           metrics=[Accuracy()],
                           scheduler_creator=ConvNetModel.scheduler_creator,
                           scheduler_step_freq="epoch",
                           use_tqdm=True,
                           backend=backend,
                           config={'data_dir': data_dir,
                                   'device': device})
est.fit(data=train_loader_func, epochs=5,
        batch_size=4, validation_data=test_loader_func)


# Step 3.3: Distributed evaluation of the trained model
result = est.evaluate(data=test_loader_func, batch_size=4)
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])


# Step 3.4: Save the trained PyTorch model
est.save("finetuned_convnet_model")


# Step 3.5: Visualize the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    train_loader, test_loader, class_names = load_dataset(config['data_dir'], batch_size)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
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


# Step 3.6: Shut down workers and releases resources
est.shutdown()


# Step 4: ConvNet as fixed feature extractor

# Here, we will freeze the weights for all of the network except that of
# the final fully connected layer. This last fully connected layer is
# replaced with a new one with random weights and only this layer is trained.


# Step 4.1: Define the model, optimizer and loss
from model import FixedConvNetModel


# Step 4.2: Finetune with Orca PyTorch Estimator
est_fixed = Estimator.from_torch(model=FixedConvNetModel.model_creator,
                                 optimizer=FixedConvNetModel.optimizer_creator,
                                 loss=nn.CrossEntropyLoss(),
                                 metrics=[Accuracy()],
                                 scheduler_creator=FixedConvNetModel.scheduler_creator,
                                 scheduler_step_freq="epoch",
                                 use_tqdm=True,
                                 backend=backend,
                                 config={'data_dir': data_dir,
                                         'device': device})
est_fixed.fit(data=train_loader_func, epochs=5,
              batch_size=4, validation_data=test_loader_func)


# Step 4.3: Distributed evaluation of the trained model
result = est_fixed.evaluate(data=test_loader_func, batch_size=4)
print('Evaluation results:')
for r in result:
    print(r, ":", result[r])


# Step 4.4: Save the trained PyTorch model
est_fixed.save("finetuned_fixed_convnet_model")


# Step 4.5: Visualize the model predictions
visualize_model(est_fixed.get_model())


# Step 4.6: Shut down workers and releases resources
est_fixed.shutdown()


# Step 5: Stop orca context
stop_orca_context()
