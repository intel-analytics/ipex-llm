/*
 * Copyright 2016 The BigDL Authors.
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
package com.intel.analytics.bigdl.orca.net

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Table
import com.intel.analytics.bigdl.orca.utils.{PythonInterpreter, PythonInterpreterTest}
import com.intel.analytics.bigdl.orca.tf.TFNetNative
import com.intel.analytics.bigdl.orca.utils.ZooSpecHelper
import com.intel.analytics.bigdl.dllib.keras.models.InternalOptimizerUtil.getStateFromOptiMethod
import com.intel.analytics.bigdl.orca.net.TorchOptim.{EpochDecay, EpochDecayByScore, IterationDecay}
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator

@PythonInterpreterTest
class TorchOptimSpec extends ZooSpecHelper{

  protected def ifskipTest(): Unit = {
    // Skip unitest if environment is not ready, PYTHONHOME should be set in environment
    if (System.getenv("PYTHONHOME") == null) {
      cancel("Please export PYTHONHOME before this test.")
    } else {
      logger.info(s"use python home: ${System.getenv("PYTHONHOME")}")
      Configurator.setLevel(PythonInterpreter.getClass().getName, Level.DEBUG)
      // Load TFNet before create interpreter, or the TFNet will throw an OMP error #13
      TFNetNative.isLoaded
    }
  }

  val lenet =
    s"""
       |import torch
       |import torch.nn as nn
       |import torch.nn.functional as F
       |from bigdl.dllib.utils.nest import ptensor_to_numpy
       |from bigdl.orca.torch import zoo_pickle_module
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
    val torchOptim = TorchOptim[Float](bys, EpochDecay)

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
    val torchOptim = TorchOptim[Float](bys, EpochDecay)

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

  "TorchOptim IterationDecay" should "work without error" in {
    ifskipTest()
    val tmpname = createTmpFile().getAbsolutePath()
    val code = lenet +
      s"""
         |model = LeNet()
         |sgd = torch.optim.SGD(model.parameters(), lr=0.1)
         |scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
         |torch.save(scheduler, "$tmpname", pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val bys = Files.readAllBytes(Paths.get(tmpname))
    val torchOptim = TorchOptim[Float](bys, IterationDecay)

    val weight = Tensor[Float](4).fill(1)
    val gradient = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    torchOptim.optimize(_ => (1f, gradient), weight)
    torchOptim.getLearningRate() should be (0.1)
    weight should be (Tensor[Float](Array(0.99f, 0.98f, 0.97f, 0.96f), Array(4)))

    val gradient2 = Tensor[Float](Array(0.2f, 0.2f, 0.2f, 0.2f), Array(4))
    torchOptim.optimize(_ => (1f, gradient2), weight)
    torchOptim.getLearningRate() should be (0.01 +- 1e-10)
    weight should be (Tensor[Float](Array(0.988f, 0.978f, 0.968f, 0.958f), Array(4)))

    val gradient3 = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    torchOptim.optimize(_ => (1f, gradient3), weight)
    torchOptim.getLearningRate() should be (0.001 +- 1e-10)
    weight should be (Tensor[Float](Array(0.9879f, 0.9778f, 0.9677f, 0.9576f), Array(4)))
  }

  "TorchOptim EpochDecay" should "work without error" in {
    ifskipTest()
    val tmpname = createTmpFile().getAbsolutePath()
    val code = lenet +
      s"""
         |model = LeNet()
         |sgd = torch.optim.SGD(model.parameters(), lr=0.1)
         |scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
         |torch.save(scheduler, "$tmpname", pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val bys = Files.readAllBytes(Paths.get(tmpname))
    val torchOptim = TorchOptim[Float](bys, EpochDecay)

    val weight = Tensor[Float](4).fill(1)
    val gradient = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    torchOptim.optimize(_ => (1f, gradient), weight)
    torchOptim.getLearningRate() should be (0.1)
    weight should be (Tensor[Float](Array(0.99f, 0.98f, 0.97f, 0.96f), Array(4)))

    val gradient2 = Tensor[Float](Array(0.2f, 0.2f, 0.2f, 0.2f), Array(4))
    torchOptim.optimize(_ => (1f, gradient2), weight)
    torchOptim.getLearningRate() should be (0.1)
    weight should be (Tensor[Float](Array(0.97f, 0.96f, 0.95f, 0.94f), Array(4)))

    val state = getStateFromOptiMethod(torchOptim)
    state("epoch") = 2

    val gradient3 = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    torchOptim.optimize(_ => (1f, gradient3), weight)
    torchOptim.getLearningRate() should be (0.01 +- 1e-10)
    weight should be (Tensor[Float](Array(0.969f, 0.958f, 0.947f, 0.936f), Array(4)))
  }

  "TorchOptim EpochDecayByScore" should "work without error" in {
    ifskipTest()
    val tmpname = createTmpFile().getAbsolutePath()
    val code = lenet +
      s"""
         |model = LeNet()
         |sgd = torch.optim.SGD(model.parameters(), lr=0.1)
         |from torch.optim.lr_scheduler import ReduceLROnPlateau
         |scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)
         |torch.save(scheduler, "$tmpname", pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val bys = Files.readAllBytes(Paths.get(tmpname))
    val torchOptim = TorchOptim[Float](bys, EpochDecayByScore)

    val weight = Tensor[Float](4).fill(1)
    val gradient = Tensor[Float](Array(0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    torchOptim.getLearningRate() should be (0.1)
    torchOptim.optimize(_ => (1f, gradient), weight)
    weight should be (Tensor[Float](Array(0.99f, 0.98f, 0.97f, 0.96f), Array(4)))
    val state = getStateFromOptiMethod(torchOptim)

    for (i <- 2 to 5) {
      state("epoch") = i
      state("score") = 0.1f
      torchOptim.optimize(_ => (1f, gradient), weight)
    }
    torchOptim.getLearningRate() should be (0.01 +- 1e-10)
    weight should be (Tensor[Float](Array(0.959f, 0.918f, 0.877f, 0.836f), Array(4)))

    val gradient2 = Tensor[Float](Array(0.1f, 0.1f, 0.1f, 0.1f), Array(4))
    for (i <- 6 to 9) {
      state("epoch") = i
      state("score") = 0.01f
      torchOptim.optimize(_ => (1f, gradient2), weight)
    }
    torchOptim.getLearningRate() should be (0.001 +- 1e-10)
    weight should be (Tensor[Float](Array(0.9559f, 0.9149f, 0.8739f, 0.8329f), Array(4)))

  }
}
