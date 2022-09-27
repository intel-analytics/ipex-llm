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


# Required Dependecies

# ```bash
# pip install neural-compressor==1.11.0 onnx onnxruntime onnxruntime_extensions
# ```


import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet18
from bigdl.nano.pytorch import Trainer
from torchmetrics import Accuracy


def finetune_pet_dataset(model_ft):

    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=.5, hue=.3),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    # Apply data augmentation to the tarin_dataset
    train_dataset = OxfordIIITPet(root="/tmp/data",
                                  transform=train_transform,
                                  download=True)
    val_dataset = OxfordIIITPet(root="/tmp/data",
                                transform=val_transform)

    # obtain training indices that will be used for validation
    indices = torch.randperm(len(train_dataset))
    val_size = len(train_dataset) // 4
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])

    # prepare data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=32)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    num_ftrs = model_ft.fc.in_features

    # Here the size of each output sample is set to 37.
    model_ft.fc = torch.nn.Linear(num_ftrs, 37)
    loss_ft = torch.nn.CrossEntropyLoss()
    optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Compile our model with loss function, optimizer.
    model = Trainer.compile(model_ft, loss_ft, optimizer_ft, metrics=[Accuracy()])
    trainer = Trainer(max_epochs=1)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return model, train_dataset, val_dataset


if __name__ == "__main__":

    model = resnet18(pretrained=True)

    _, train_dataset, val_dataset = finetune_pet_dataset(model)

    # Sample Inference Data
    x = torch.stack([val_dataset[0][0], val_dataset[1][0]])

    # Normal Inference
    model.eval()
    y_hat = model(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)

    # Static Quantization for ONNX
    from bigdl.nano.pytorch import InferenceOptimizer
    q_model = InferenceOptimizer.quantize(model,
                                 accelerator='onnxruntime',
                                 calib_dataloader=DataLoader(train_dataset, batch_size=32))

    # Inference with Quantized Model
    y_hat = q_model(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)

    # Save Quantized Model
    Trainer.save(q_model, "./quantized_model")

    # Load the Quantized Model
    # loaded_model = Trainer.load("./quantized_model")
    # print(loaded_model(x).argmax(dim=1))
