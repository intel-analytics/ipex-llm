#!/usr/bin/env python
# coding: utf-8

# # Apply ONNXRuntime Acceleration on Inference Pipeline

# ### Environment Preparation

# ```bash
# pip install onnx onnxruntime
# ```

import torch

if __name__ == "__main__":

    import torch
    from torchvision.models import resnet18
    model_ft = resnet18(pretrained=True)

    # Normal Inference
    x = torch.stack(torch.rand(1, 3, 224, 224))
    model_ft.eval()
    y_hat = model_ft(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)

    # Accelerated Inference Using Onnxruntime
    from bigdl.nano.pytorch import Trainer
    ort_model = Trainer.trace(model_ft,
                              accelerator="onnxruntime",
                              input_sample=torch.rand(1, 3, 224, 224))

    y_hat = ort_model(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)
