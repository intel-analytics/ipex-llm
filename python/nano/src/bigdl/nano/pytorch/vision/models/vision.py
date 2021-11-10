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
#


import torchvision
from ._utils import *


def resnet18(pretrained: bool = False, include_top: bool = False, freeze: bool = False):
    """
    Create an instance of resnet18 model.

    :param pretrained: (bool) Whether to return a model pre-trained on ImageNet. Defaults to False.
    :param include_top: (bool) Whether to include the fully-connected layer at the top of the
           network. If False, it will be replaced by an identity layer, and the output size
           of the returned model can be obtained by `model.get_output_size()`. Defaults to False.
    :param freeze: (bool) Whether to freeze the model besides the top layer of the network (only
           enable training on the top layer). If `include_top` is False and `freeze` is True,
           then all layers will be frozen in the model to be returned. Defaults to False.
    :return: PyTorch model resnet18.
    """

    model = torchvision.models.resnet18(pretrained=pretrained)
    if freeze:
        set_parameter_requires_grad(model, "fc")
    if not include_top:
        output_size = model.fc.in_features
        model.fc = nn.Identity()
        return BackboneModule(model, output_size)
    else:
        return model


def resnet34(pretrained: bool = False, include_top: bool = False, freeze: bool = False):
    """
    Create an instance of resnet34 model.

    :param pretrained: (bool) Whether to return a model pre-trained on ImageNet. Defaults to False.
    :param include_top: (bool) Whether to include the fully-connected layer at the top of the
           network. If False, it will be replaced by an identity layer, and the output size
           of the returned model can be obtained by `model.get_output_size()`. Defaults to False.
    :param freeze: (bool) Whether to freeze the model besides the top layer of the network (only
           enable training on the top layer). If `include_top` is False and `freeze` is True,
           then all layers will be frozen in the model to be returned. Defaults to False.
    :return: PyTorch model resnet34.
    """

    model = torchvision.models.resnet34(pretrained)
    if freeze:
        set_parameter_requires_grad(model, "fc")
    if not include_top:
        output_size = model.fc.in_features
        model.fc = nn.Identity()
        return BackboneModule(model, output_size)
    else:
        return model


def resnet50(pretrained: bool = False, include_top: bool = False, freeze: bool = False):
    """
    Create an instance of resnet50 model.

    :param pretrained: (bool) Whether to return a model pre-trained on ImageNet. Defaults to False.
    :param include_top: (bool) Whether to include the fully-connected layer at the top of the
           network. If False, it will be replaced by an identity layer, and the output size
           of the returned model can be obtained by `model.get_output_size()`. Defaults to False.
    :param freeze: (bool) Whether to freeze the model besides the top layer of the network (only
           enable training on the top layer). If `include_top` is False and `freeze` is True,
           then all layers will be frozen in the model to be returned. Defaults to False.
    :return: PyTorch model resnet50.
    """

    model = torchvision.models.resnet50(pretrained)
    if freeze:
        set_parameter_requires_grad(model, "fc")
    if not include_top:
        output_size = model.fc.in_features
        model.fc = nn.Identity()
        return BackboneModule(model, output_size)
    else:
        return model


def mobilenet_v3_large(pretrained: bool = False, include_top: bool = False, freeze: bool = False):
    """
    Create an instance of mobilenet_v3_large model.

    :param pretrained: (bool) Whether to return a model pre-trained on ImageNet. Defaults to False.
    :param include_top: (bool) Whether to include the fully-connected layer at the top of the
           network. If False, it will be replaced by an identity layer, and the output size
           of the returned model can be obtained by `model.get_output_size()`. Defaults to False.
    :param freeze: (bool) Whether to freeze the model besides the top layer of the network (only
           enable training on the top layer). If `include_top` is False and `freeze` is True,
           then all layers will be frozen in the model to be returned. Defaults to False.
    :return: PyTorch model mobilenet_v3_large.
    """

    model = torchvision.models.mobilenet_v3_large(pretrained)
    if freeze:
        set_parameter_requires_grad(model, "classifier")
    if not include_top:
        output_size = model.classifier[0].in_features
        model.classifier = nn.Identity()
        return BackboneModule(model, output_size)
    else:
        return model


def mobilenet_v3_small(pretrained: bool = False, include_top: bool = False, freeze: bool = False):
    """
    Create an instance of mobilenet_v3_small model.

    :param pretrained: (bool) Whether to return a model pre-trained on ImageNet. Defaults to False.
    :param include_top: (bool) Whether to include the fully-connected layer at the top of the
           network. If False, it will be replaced by an identity layer, and the output size
           of the returned model can be obtained by `model.get_output_size()`. Defaults to False.
    :param freeze: (bool) Whether to freeze the model besides the top layer of the network (only
           enable training on the top layer). If `include_top` is False and `freeze` is True,
           then all layers will be frozen in the model to be returned. Defaults to False.
    :return: PyTorch model mobilenet_v3_small.
    """

    model = torchvision.models.mobilenet_v3_small(pretrained)
    if freeze:
        set_parameter_requires_grad(model, "classifier")
    if not include_top:
        output_size = model.classifier[0].in_features
        model.classifier = nn.Identity()
        return BackboneModule(model, output_size)
    else:
        return model


def mobilenet_v2(pretrained: bool = False, include_top: bool = False, freeze: bool = False):
    """
    Create an instance of mobilenet_v2 model.

    :param pretrained: (bool) Whether to return a model pre-trained on ImageNet. Defaults to False.
    :param include_top: (bool) Whether to include the fully-connected layer at the top of the
           network. If False, it will be replaced by an identity layer, and the output size
           of the returned model can be obtained by `model.get_output_size()`. Defaults to False.
    :param freeze: (bool) Whether to freeze the model besides the top layer of the network (only
           enable training on the top layer). If `include_top` is False and `freeze` is True,
           then all layers will be frozen in the model to be returned. Defaults to False.
    :return: PyTorch model mobilenet_v2.
    """

    model = torchvision.models.mobilenet_v2(pretrained)
    if freeze:
        set_parameter_requires_grad(model, "classifier")
    if not include_top:
        output_size = model.classifier[1].in_features
        model.classifier = nn.Identity()
        return BackboneModule(model, output_size)
    else:
        return model


def shufflenet_v2_x1_0(pretrained: bool = False, include_top: bool = False, freeze: bool = False):
    """
    Create an instance of shufflenet_v2_x1_0 model.

    :param pretrained: (bool) Whether to return a model pre-trained on ImageNet. Defaults to False.
    :param include_top: (bool) Whether to include the fully-connected layer at the top of the
           network. If False, it will be replaced by an identity layer, and the output size
           of the returned model can be obtained by `model.get_output_size()`. Defaults to False.
    :param freeze: (bool) Whether to freeze the model besides the top layer of the network (only
           enable training on the top layer). If `include_top` is False and `freeze` is True,
           then all layers will be frozen in the model to be returned. Defaults to False.
    :return: PyTorch model shufflenet_v2_x1_0.
    """

    model = torchvision.models.shufflenet_v2_x1_0(pretrained)
    if freeze:
        set_parameter_requires_grad(model, "fc")
    if not include_top:
        output_size = model.fc.in_features
        model.fc = nn.Identity()
        return BackboneModule(model, output_size)
    else:
        return model
