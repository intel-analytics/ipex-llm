#
# Copyright 2018 Analytics Zoo Authors.
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
#

from optparse import OptionParser

import torch
from torchvision import datasets, models, transforms
from zoo.common.nncontext import init_nncontext
from zoo.feature.common import *
from zoo.feature.image import *
from zoo.pipeline.api.net.torch_net import TorchNet


def predict(img_path):
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])),
        batch_size=8, shuffle=False,
        num_workers=1, pin_memory=True)

    model = models.resnet18(pretrained=True).eval()
    net = TorchNet.from_pytorch(model, [1, 3, 224, 224])

    for inputs, labels in val_loader:
        output = net.predict(inputs.numpy(), distributed=True).collect()
        index = [o.argmax() for o in output]
        print(index)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--image", type=str, dest="img_path",
                      help="The path where the images are stored, "
                           "can be either a folder or an image path")
    parser.add_option("--model", type=str, dest="model_path",
                      help="The path of the TensorFlow object detection model")
    parser.add_option("--partition_num", type=int, dest="partition_num", default=4,
                      help="The number of partitions")
    (options, args) = parser.parse_args(sys.argv)

    sc = init_nncontext("Torch ResNet Prediction Example")
    predict(options.img_path)
