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

import com.intel.analytics.bigdl.dllib.feature.dataset.Sample
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{Engine, RandomGenerator}
import com.intel.analytics.bigdl.orca.utils.{PythonInterpreter, PythonInterpreterTest}
import com.intel.analytics.bigdl.orca.tf.TFNetNative
import com.intel.analytics.bigdl.orca.utils.ZooSpecHelper
import jep.NDArray
import org.apache.spark.{SparkConf, SparkContext}
import com.intel.analytics.bigdl.dllib.NNContext.initNNContext
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator

@PythonInterpreterTest
class TorchModelSpec extends ZooSpecHelper{

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
         |byc = CloudPickleSerializer.dumps(CloudPickleSerializer, criterion)
         |bys = io.BytesIO()
         |torch.save(model, bys, pickle_module=zoo_pickle_module)
         |bym = bys.getvalue()
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
        |_data = (input, target)
        |""".stripMargin
    PythonInterpreter.exec(genInputCode)
    val output = model.forward(Tensor[Float]())
    criterion.forward(output, Tensor[Float]())
    criterion.backward(output, Tensor[Float]())
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
         |bys = io.BytesIO()
         |torch.save(model, bys, pickle_module=zoo_pickle_module)
         |bym = bys.getvalue()
         |""".stripMargin
    PythonInterpreter.exec(code)
    val w = PythonInterpreter.getValue[NDArray[Array[Float]]]("flatten_weight").getData()
    val bys = PythonInterpreter.getValue[Array[Byte]]("bym")
    val model = TorchModel(bys, w)

    val genInputCode =
      """
        |import numpy as np
        |input = torch.tensor(np.random.rand(4, 1, 28, 28), dtype=torch.float32)
        |_data = [input]
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
         |def lossFunc(input, target):
         |    return nn.CrossEntropyLoss().forward(input, target.flatten().long())
         |from pyspark.serializers import CloudPickleSerializer
         |weights=[]
         |for param in model.parameters():
         |    weights.append(param.view(-1))
         |flatten_weight = torch.nn.utils.parameters_to_vector(weights).data.numpy()
         |byc = CloudPickleSerializer.dumps(CloudPickleSerializer, lossFunc)
         |bys = io.BytesIO()
         |torch.save(model, bys, pickle_module=zoo_pickle_module)
         |bym = bys.getvalue()
         |if '_data' in locals():
         |  del _data
         |""".stripMargin
    PythonInterpreter.exec(code)

    val w = PythonInterpreter.getValue[NDArray[Array[Float]]]("flatten_weight").getData()
    val bys = PythonInterpreter.getValue[Array[Byte]]("bym")
    val model = TorchModel(bys, w)
    val c = PythonInterpreter.getValue[Array[Byte]]("byc")
    val criterion = TorchLoss(c)

    val input = Tensor[Float](4, 1, 28, 28).rand()
    val target = Tensor[Float](Array(0f, 1f, 3f, 4f), Array(4))
    val output = model.forward(input)
    criterion.forward(output, target)
    val gradOutput = criterion.backward(output, target)
    model.backward(input, gradOutput)
  }

  "TorchModel's get/set param" should "works" in {
    ifskipTest()
    val code = lenet +
      s"""
         |import torch
         |import torchvision
         |model = torchvision.models.resnet18()
         |from pyspark.serializers import CloudPickleSerializer
         |weights=[]
         |for param in model.parameters():
         |    weights.append(param.view(-1))
         |flatten_weight = torch.nn.utils.parameters_to_vector(weights).data.numpy()
         |bys = io.BytesIO()
         |torch.save(model, bys, pickle_module=zoo_pickle_module)
         |bym = bys.getvalue()
         |""".stripMargin
    PythonInterpreter.exec(code)

    val w = PythonInterpreter.getValue[NDArray[Array[Float]]]("flatten_weight").getData()
    val bys = PythonInterpreter.getValue[Array[Byte]]("bym")
    val model = TorchModel(bys, w)
    model.training()
    val extraParams = model.getExtraParameter()
    extraParams.foreach{v =>
      if (v.isScalar) {
        v.fill(3)
      } else {
        v.rand()
      }
    }
    model.setExtraParam(extraParams)
    val newExtraParams = model.getExtraParameter()
    extraParams.zip(newExtraParams).foreach{v =>
      v._1 should be (v._2)
    }
  }

  "TorchModel" should "loadModel without error" in {
    ifskipTest()
    val tmpname = createTmpFile().getAbsolutePath()
    val code = lenet +
      s"""
         |model = LeNet()
         |torch.save(model, "$tmpname", pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val model = TorchModel.loadModel(tmpname)
    model.evaluate()

    val genInputCode =
      s"""
         |import numpy as np
         |import torch
         |input = torch.tensor(np.random.rand(4, 1, 28, 28), dtype=torch.float32)
         |target = torch.tensor(np.ones([4]), dtype=torch.long)
         |_data = (input, target)
         |""".stripMargin
    PythonInterpreter.exec(genInputCode)
    model.forward(Tensor[Float]())
  }

  "SimpleTorchModel" should "predict without error" in {
    ifskipTest()
    val conf = Engine.createSparkConf()
      .setAppName("SimpleTorchModel").setMaster("local[4]")
    val sc = initNNContext(conf)
    val tmpname = createTmpFile().getAbsolutePath()
    val code =
      s"""
         |import torch
         |from torch import nn
         |from bigdl.orca.torch import zoo_pickle_module
         |if '_data' in locals():
         |  del _data
         |
         |class SimpleTorchModel(nn.Module):
         |    def __init__(self):
         |        super(SimpleTorchModel, self).__init__()
         |        self.dense1 = nn.Linear(2, 1)
         |        list(self.dense1.parameters())[0][0][0] = 0.2
         |        list(self.dense1.parameters())[0][0][1] = 0.5
         |        list(self.dense1.parameters())[1][0] = 0.3
         |    def forward(self, x):
         |        x = self.dense1(x)
         |        return x
         |
         |model = SimpleTorchModel()
         |torch.save(model, "$tmpname", pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val model = TorchModel.loadModel(tmpname)
    model.evaluate()
    RandomGenerator.RNG.setSeed(1L)
    val input = Array.tabulate(1024)(_ =>
      (RandomGenerator.RNG.uniform(0, 1).toFloat,
        RandomGenerator.RNG.uniform(0, 1).toFloat))
    val exceptedTarget = input.map(v => 0.2f * v._1 + 0.5f * v._2 + 0.3f)
    val rddSample = sc.parallelize(input, 4).map(v =>
      Sample(Tensor[Float](Array(v._1, v._2), Array(2))))
    val results = model.predict(rddSample, batchSize = 16).collect()
    (0 until 1024).foreach{i =>
      results(i).toTensor[Float].value() should be (exceptedTarget(i))
    }
    sc.stop()
  }

  "SimpleTorchModel" should "predict without error with multioutput" in {
    ifskipTest()
    val conf = Engine.createSparkConf()
      .setAppName("SimpleTorchModel").setMaster("local[4]")
    val sc = initNNContext(conf)
    val tmpname = createTmpFile().getAbsolutePath()
    val code =
      s"""
         |import torch
         |from torch import nn
         |from bigdl.orca.torch import zoo_pickle_module
         |if '_data' in locals():
         |  del _data
         |
         |class SimpleTorchModel(nn.Module):
         |    def __init__(self):
         |        super(SimpleTorchModel, self).__init__()
         |        self.dense1 = nn.Linear(2, 1)
         |        list(self.dense1.parameters())[0][0][0] = 0.2
         |        list(self.dense1.parameters())[0][0][1] = 0.5
         |        list(self.dense1.parameters())[1][0] = 0.3
         |    def forward(self, x):
         |        x1 = self.dense1(x)
         |        x2 = self.dense1(x)
         |        return (x1, x2)
         |
         |model = SimpleTorchModel()
         |torch.save(model, "$tmpname", pickle_module=zoo_pickle_module)
         |""".stripMargin
    PythonInterpreter.exec(code)
    val model = TorchModel.loadModel(tmpname)
    model.evaluate()
    RandomGenerator.RNG.setSeed(1L)
    val input = Array.tabulate(1024)(_ =>
      (RandomGenerator.RNG.uniform(0, 1).toFloat,
        RandomGenerator.RNG.uniform(0, 1).toFloat))
    val exceptedTarget = input.map(v => 0.2f * v._1 + 0.5f * v._2 + 0.3f)
    val rddSample = sc.parallelize(input, 4).map(v =>
      Sample(Tensor[Float](Array(v._1, v._2), Array(2))))
    val results = model.predict(rddSample, batchSize = 16).collect()
    (0 until 1024).foreach{i =>
      results(i).toTable[Tensor[Float]](1).value() should be (exceptedTarget(i))
      results(i).toTable[Tensor[Float]](2).value() should be (exceptedTarget(i))
    }
    sc.stop()
  }
}
