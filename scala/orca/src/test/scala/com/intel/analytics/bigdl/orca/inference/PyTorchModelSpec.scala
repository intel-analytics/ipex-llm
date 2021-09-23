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

package com.intel.analytics.bigdl.orca.inference

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.orca.utils.{PythonInterpreter, PythonInterpreterTest}
import com.intel.analytics.zoo.core.TFNetNative
import com.intel.analytics.bigdl.orca.utils.ZooSpecHelper
import org.apache.log4j.{Level, Logger}

import scala.language.postfixOps


@PythonInterpreterTest
class PyTorchModelSpec extends ZooSpecHelper with InferenceSupportive {

  var model: InferenceModel = _
  var model2: InferenceModel = _
  val currentNum = 10
  var modelPath: String = _

  override def doBefore(): Unit = {
    model = new InferenceModel(currentNum) { }
    model2 = new InferenceModel(currentNum) { }
    modelPath = ZooSpecHelper.createTmpFile().getAbsolutePath()
  }

  override def doAfter(): Unit = {
    model.doRelease()
    model2.doRelease()
  }

  protected def ifskipTest(): Unit = {
    // Skip unitest if environment is not ready, PYTHONHOME should be set in environment
    if (System.getenv("PYTHONHOME") == null) {
      cancel("Please export PYTHONHOME before this test.")
    } else {
      logger.info(s"use python home: ${System.getenv("PYTHONHOME")}")
      Logger.getLogger(PythonInterpreter.getClass()).setLevel(Level.DEBUG)
      // Load TFNet before create interpreter, or the TFNet will throw an OMP error #13
      TFNetNative.isLoaded
      val resnetModel =
        s"""
           |import torch
           |import torch.nn as nn
           |import torchvision.models as models
           |from zoo.pipeline.api.torch import zoo_pickle_module
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
           |torch.save(model, "$modelPath", pickle_module=zoo_pickle_module)
           |""".stripMargin
      PythonInterpreter.exec(resnetModel)
    }
  }


  "PyTorch Model" should "be loaded" in {
    ifskipTest()
    val pyTorchModel = ModelLoader.loadFloatModelForPyTorch(modelPath)
    pyTorchModel.evaluate()
    val metaModel = makeMetaModel(pyTorchModel)
    val floatFromPyTorch = new FloatModel(pyTorchModel, metaModel, true)
    floatFromPyTorch shouldNot be(null)

    model.doLoadPyTorch(modelPath)
    model shouldNot be(null)

    val modelBytes = Files.readAllBytes(Paths.get(modelPath))
    model2.doLoadPyTorch(modelBytes)
    model2 shouldNot be(null)

    (0 until currentNum).toParArray.foreach(i => {
      model.doLoadPyTorch(modelPath)
      model shouldNot be(null)

      model2.doLoadPyTorch(modelBytes)
      model2 shouldNot be(null)
      1f
    })

  }

  "PyTorch Model" should "do predict" in {
    ifskipTest()
    model.doLoadPyTorch(modelPath)
    model2.doLoadPyTorch(modelPath)

    (0 until currentNum * 10).toParArray.foreach(i => {
      val inputTensor = Tensor[Float](1, 2).rand()
      val exceptedResult = inputTensor.valueAt(1, 1) * 0.2f +
        inputTensor.valueAt(1, 2) * 0.5f + 0.3f
      val r = model.doPredict(inputTensor)
      r should be(Tensor[Float](Array(exceptedResult), Array(1, 1)))

      val r2 = model2.doPredict(inputTensor)
      r2 should be(Tensor[Float](Array(exceptedResult), Array(1, 1)))
      1f
    })

  }

}
