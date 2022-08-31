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

import torch
from torchmetrics import Accuracy
from _finetune import MilestonesFinetuning, TransferLearningModel, CatDogImageDataModule

from bigdl.nano.pytorch.trainer import Trainer
from bigdl.nano.pytorch import InferenceOptimizer


if __name__ == "__main__":
    # First finetune on new dataset
    milestones: tuple = (5, 10)
    trainer = Trainer(max_epochs=15, callbacks=[MilestonesFinetuning(milestones)])
    model = TransferLearningModel(milestones=milestones)
    datamodule = CatDogImageDataModule()
    trainer.fit(model, datamodule)

    # define metric for accuracy calculation
    def accuracy(pred, target):
        pred = torch.sigmoid(pred)
        target = target.view((-1, 1)).type_as(pred).int()
        return Accuracy()(pred, target)

    # accelaration inference using InferenceOptimizer
    model.eval()
    optimizer = InferenceOptimizer()
    # optimize may take about 10 minutes to run all possible accelaration combinations
    optimizer.optimize(model=model,
                       training_data=datamodule.train_dataloader(batch_size=1),
                       validation_data=datamodule.val_dataloader(),
                       metric=accuracy,
                       direction="max",
                       cpu_num=1,
                       latency_sample_num=30)

    for key, value in optimizer.optimized_model_dict.items():
        print("accleration option: {}, latency: {:.4f}ms, accuracy: {:.4f}".format(key, value["latency"], value["accuracy"]))

    acc_model, option = optimizer.get_best_model(accelerator="onnxruntime")
    print("When accelerator is onnxruntime, the model with the least latency is: ", option)

    acc_model, option = optimizer.get_best_model(accuracy_criterion=0.05)
    print("When accuracy drop less than 5%, the model with the least latency is: ", option)

    acc_model, option = optimizer.get_best_model()
    print("The model with the least latency is: ", option)
