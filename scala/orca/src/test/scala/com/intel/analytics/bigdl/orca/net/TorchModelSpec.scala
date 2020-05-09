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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.{PythonInterpreter, PythonInterpreterTest}
import com.intel.analytics.zoo.core.TFNetNative
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import jep.NDArray
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

@PythonInterpreterTest
class TorchModelSpec extends ZooSpecHelper{

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

  "TorchModel" should "forward without error" in {
    ifskipTest()
    val code = lenet +
      s"""
         |model = LeNet()
         |criterion = nn.CrossEntropyLoss()
         |from pyspark.serializers import CloudPickleSerializer
         |weights=[]
         |for param in model.parameters():
         |    weights.append(param.view(-1))
         |flatten_weight = torch.nn.utils.parameters_to_vector(weights).data.numpy()
         |bym = CloudPickleSerializer.dumps(CloudPickleSerializer, model)
         |byc = CloudPickleSerializer.dumps(CloudPickleSerializer, criterion)
         |""".stripMargin
    PythonInterpreter.exec(code)

    val w = PythonInterpreter.getValue[NDArray[Array[Float]]]("flatten_weight").getData()
    val bys = PythonInterpreter.getValue[Array[Byte]]("bym")
    val model = TorchModel(bys, w)
    val c = PythonInterpreter.getValue[Array[Byte]]("byc")
    val criterion = TorchLoss(c)

    val genInputCode =
      """
        |import numpy as np
        |input = torch.tensor(np.random.rand(4, 1, 28, 28), dtype=torch.float32)
        |target = torch.tensor(np.ones([4]), dtype=torch.long)
        |data = (input, target)
        |""".stripMargin
    PythonInterpreter.exec(genInputCode)
    model.forward(Tensor[Float]())
    criterion.forward(Tensor[Float](), Tensor[Float]())
    criterion.backward(Tensor[Float](), Tensor[Float]())
    model.backward(Tensor[Float](), Tensor[Float]())
  }

  "setWeights" should "works fine" in {
    ifskipTest()
    val code = lenet +
      s"""
         |model = LeNet()
         |from pyspark.serializers import CloudPickleSerializer
         |weights=[]
         |for param in model.parameters():
         |    weights.append(param.view(-1))
         |flatten_weight = torch.nn.utils.parameters_to_vector(weights).data.numpy()
         |bym = CloudPickleSerializer.dumps(CloudPickleSerializer, model)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val w = PythonInterpreter.getValue[NDArray[Array[Float]]]("flatten_weight").getData()
    val bys = PythonInterpreter.getValue[Array[Byte]]("bym")
    val model = TorchModel(bys, w)

    val genInputCode =
      """
        |import numpy as np
        |input = torch.tensor(np.random.rand(4, 1, 28, 28), dtype=torch.float32)
        |data = [input]
        |""".stripMargin
    PythonInterpreter.exec(genInputCode)
    val output1 = model.forward(Tensor[Float]())
    output1 should not be (Tensor[Float](4, 10).fill(-2.3025851f))

    // fill all weight and bias to zero, then fc's output should be zero,
    // the LogSoftMax's result should be all -2.3025851
    model.weights.fill(0)
    val output2 = model.forward(Tensor[Float]())
    output2 should be (Tensor[Float](4, 10).fill(-2.3025851f))
  }

  "TorchModel" should "forward using scala input and target" in {
    ifskipTest()
    val code = lenet +
      s"""
         |model = LeNet()
         |criterion = nn.CrossEntropyLoss()
         |from pyspark.serializers import CloudPickleSerializer
         |weights=[]
         |for param in model.parameters():
         |    weights.append(param.view(-1))
         |flatten_weight = torch.nn.utils.parameters_to_vector(weights).data.numpy()
         |bym = CloudPickleSerializer.dumps(CloudPickleSerializer, model)
         |byc = CloudPickleSerializer.dumps(CloudPickleSerializer, criterion)
         |del data
         |""".stripMargin
    PythonInterpreter.exec(code)

    val w = PythonInterpreter.getValue[NDArray[Array[Float]]]("flatten_weight").getData()
    val bys = PythonInterpreter.getValue[Array[Byte]]("bym")
    val model = TorchModel(bys, w)
    val c = PythonInterpreter.getValue[Array[Byte]]("byc")
    val criterion = TorchLoss(c)

    val input = Tensor[Float](4, 1, 28, 28).rand()
    val target = Tensor[Float](Array(0f, 1f, 3f, 4f), Array(4))
    model.forward(input)
    criterion.forward(input, target)
    val gradOutput = criterion.backward(input, target)
    model.backward(input, gradOutput)
  }
}
