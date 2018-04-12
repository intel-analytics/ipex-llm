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
package com.intel.analytics.zoo.pipeline.api.keras

import java.io.{File => JFile}

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.zoo.models.common.ZooModel
import org.apache.log4j.Logger
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer


abstract class ZooSpecHelper extends FlatSpec with Matchers with BeforeAndAfter {
  protected val logger = Logger.getLogger(getClass)

  private val tmpFiles : ArrayBuffer[JFile] = new ArrayBuffer[JFile]()

  protected def createTmpFile(): JFile = {
    val file = java.io.File.createTempFile("UnitTest", "BigDLSpecBase")
    logger.info(s"created file $file")
    tmpFiles.append(file)
    file
  }

  protected def getFileFolder(path: String): String = {
    path.substring(0, path.lastIndexOf(JFile.separator))
  }

  protected def getFileName(path: String): String = {
    path.substring(path.lastIndexOf(JFile.separator) + 1)
  }

  def doAfter(): Unit = {}

  def doBefore(): Unit = {}

  before {
    doBefore()
  }

  after {
    doAfter()
    tmpFiles.foreach(f => {
      if (f.exists()) {
        require(f.isFile, "cannot clean folder")
        f.delete()
        logger.info(s"deleted file $f")
      }
    })
  }

  def compareOutputAndGradInput(model1: AbstractModule[Tensor[Float], Tensor[Float], Float],
                                model2: AbstractModule[Tensor[Float], Tensor[Float], Float],
                                input: Tensor[Float],
                                precision: Double = 1e-5): Unit = {
    // Set seed in case of random factors such as dropout
    RandomGenerator.RNG.setSeed(1000)
    val toutput = model1.forward(input)
    RandomGenerator.RNG.setSeed(1000)
    val koutput = model2.forward(input)
    koutput.almostEqual(toutput, precision) should be (true)
    RandomGenerator.RNG.setSeed(1000)
    val tgradInput = model1.backward(input, toutput)
    RandomGenerator.RNG.setSeed(1000)
    val kgradInput = model2.backward(input, koutput)
    kgradInput.almostEqual(tgradInput, precision) should be (true)
  }

  def compareOutputAndGradInputSetWeights(
      model1: AbstractModule[Tensor[Float], Tensor[Float], Float],
      model2: AbstractModule[Tensor[Float], Tensor[Float], Float],
      input: Tensor[Float],
      precision: Double = 1e-5): Unit = {
    if (model1.getWeightsBias() != null) {
      model2.setWeightsBias(model1.getWeightsBias())
    }
    compareOutputAndGradInput(model1, model2, input, precision)
  }

  def testZooModelLoadSave(model: AbstractModule[Tensor[Float], Tensor[Float], Float],
                           input: Tensor[Float],
                           loader: (String, String) => AbstractModule[Activity, Activity, Float],
                           precision: Double = 1e-5): Unit = {
    val serFile = JFile.createTempFile(model.getName(), "model")
    model.saveModule(serFile.getAbsolutePath, overWrite = true)
    val loadedModel = loader(serFile.getAbsolutePath, null)
      .asInstanceOf[ZooModel[Activity, Activity, Float]]
    require(loadedModel.modules.length == 1)
    compareOutputAndGradInput(model,
      loadedModel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      input, precision)
    if (serFile.exists) {
      serFile.delete
    }
  }
}
