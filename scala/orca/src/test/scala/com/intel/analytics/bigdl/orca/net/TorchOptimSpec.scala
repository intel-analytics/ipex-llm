/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.zoo.pipeline.api.net

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.zoo.common.{PythonInterpreter, PythonInterpreterTest}
import com.intel.analytics.zoo.core.TFNetNative
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.models.InternalOptimizerUtil.getStateFromOptiMethod
import org.apache.log4j.{Level, Logger}

@PythonInterpreterTest
class TorchOptimSpec extends ZooSpecHelper{

  protected def ifskipTest(): Unit = {
    // Skip unitest if environment is not ready, PYTHONHOME should be set in environment
    if (System.getenv("PYTHONHOME") == null) {
      cancel("Please export PYTHONHOME before this test.")
    } else {
      logger.info(s"use python home: ${System.getenv("PYTHONHOME")}")
      Logger.getLogger(PythonInterpreter.getClass()).setLevel(Level.DEBUG)
      // Load TFNet before create interpreter, or the TFNet will throw an OMP error #13
      TFNetNative.isLoaded
    }
  }

  val lenet =
    s"""
       |import torch
       |import torch.nn as nn
       |import torch.nn.functional as F
       |from zoo.util.nest import ptensor_to_numpy
       |from zoo.pipeline.api.torch import zoo_pickle_module
       |import io
       |
       |class LeNet(nn.Module):
       |    def __init__(self):
       |        super(LeNet, self).__init__()
       |        self.conv1 = nn.Conv2d(1, 20, 5, 1)
       |        self.conv2 = nn.Conv2d(20, 50, 5, 1)
       |        self.fc1 = nn.Linear(4*4*50, 500)
       |        self.fc2 = nn.Linear(500, 10)
       |    def forward(self, x):
       |        x = F.relu(self.conv1(x))
       |        x = F.max_pool2d(x, 2, 2)
       |        x = F.relu(self.conv2(x))
       |        x = F.max_pool2d(x, 2, 2)
       |        x = x.view(-1, 4*4*50)
       |        x = F.relu(self.fc1(x))
       |        x = self.fc2(x)
       |        return F.log_softmax(x, dim=1)
       |""".stripMargin

  "TorchOptim" should "load without error" in {
    ifskipTest()
    val tmpname = createTmpFile().getAbsolutePath()
    val code = lenet +
      s"""
         |model = LeNet()
         |optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
         |torch.save(optimizer, "$tmpname", pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val bys = Files.readAllBytes(Paths.get(tmpname))
    val torchOptim = TorchOptim[Float](bys)

    val weight = Tensor[Float](4).fill(1)
    val gradient = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    torchOptim.optimize(_ => (1f, gradient), weight)
    weight should be (Tensor[Float](Array(0.99f, 0.98f, 0.97f, 0.96f), Array(4)))
  }

  "TorchOptim" should "load lrscheduler without error" in {
    ifskipTest()
    val tmpname = createTmpFile().getAbsolutePath()
    val code = lenet +
      s"""
         |model = LeNet()
         |lambda1 = lambda epoch: 0.95 ** epoch
         |sgd = torch.optim.SGD(model.parameters(), lr=0.1)
         |from torch.optim.lr_scheduler import LambdaLR
         |scheduler = LambdaLR(sgd, lr_lambda=[lambda1])
         |torch.save(scheduler, "$tmpname", pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val bys = Files.readAllBytes(Paths.get(tmpname))
    val torchOptim = TorchOptim[Float](bys)

    val weight = Tensor[Float](4).fill(1)
    val gradient = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    torchOptim.optimize(_ => (1f, gradient), weight)
    torchOptim.getLearningRate() should be (0.1)
    weight should be (Tensor[Float](Array(0.99f, 0.98f, 0.97f, 0.96f), Array(4)))
    val gradient2 = Tensor[Float](Array(0.2f, 0.2f, 0.2f, 0.2f), Array(4))
    val state = getStateFromOptiMethod(torchOptim)
    state("epoch") = 2
    torchOptim.optimize(_ => (1f, gradient2), weight)
    torchOptim.getLearningRate() should be (0.095)
    weight should be (Tensor[Float](Array(0.971f, 0.961f, 0.951f, 0.941f), Array(4)))
  }
}
