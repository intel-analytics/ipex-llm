/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.python


import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.ppml.algorithms.PSI
import com.intel.analytics.bigdl.ppml.algorithms.vfl.FGBoostRegression
import com.intel.analytics.bigdl.ppml.{FLClient, FLContext, FLModel, FLServer}
import com.intel.analytics.bigdl.ppml.fgboost.FGBoostModel
import com.intel.analytics.bigdl.ppml.pytorch.PytorchSuite
import com.intel.analytics.bigdl.ppml.utils.FLClientClosable

import java.util.{List => JList}
import scala.collection.JavaConverters._
import scala.reflect.ClassTag


object PythonPPML {

  def ofFloat(): PythonPPML[Float] = new PythonPPML[Float]()

  def ofDouble(): PythonPPML[Double] = new PythonPPML[Double]()
}
class PythonPPML[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL {
  def initFLContext() = {
    FLContext.initFLContext()
  }
  def createFLServer() = {
    new FLServer()
  }
  def createHflNN() = {

  }
  def createHflLogisticRegression() = {

  }
  def createHflLinearRegression() = {

  }
  def createVflLogisticRegression() = {

  }
  def createVflLinearRegression() = {

  }

  def createFGBoostRegression(learningRate: Double, maxDepth: Int, minChildSize: Int) = {
    new FGBoostRegression(learningRate.toFloat, maxDepth, minChildSize)
  }
  def createFGBoostClassification() = {

  }
  def flServerBuild(flServer: FLServer) = {
    flServer.build()
  }
  def flServerStart(flServer: FLServer) = {
    flServer.start()
  }
  def flServerStop(flServer: FLServer) = {
    flServer.stop()
  }

  def flServerSetClientNum(flServer: FLServer, clientNum: Int) = {
    flServer.setClientNum(clientNum)
  }

  /**
   * FlClient is not exposed to users API, the Python API for this only locates in test
   * @param target the FlClient target Url
   * @return
   */
  def createFLClient(target: String) = {
    val flClient = new FLClient()
    if (target != null) flClient.setTarget(target)
    flClient
  }
  def flClientClosableSetFLClient(flClientClosable: FLClientClosable, flClient: FLClient) = {
    flClientClosable.setFlClient(flClient)
  }
  def createPSI() = {
    new PSI()
  }
  def psiGetSalt(psi: PSI, secureCode: String = ""): String = {
    psi.getSalt(secureCode)
  }
  def psiUploadSet(psi: PSI, ids: JList[String], salt: String) = {
    psi.uploadSet(ids, salt)
  }
  def psiDownloadIntersection(psi: PSI, maxtry: Int = 100, retry: Int = 3000) = {
    psi.downloadIntersection(maxtry, retry)
  }
  def jTensorToTensorArray(jTensor: JTensor) = {
    require(jTensor.shape.length == 2, "FGBoost only support 2D input")
    val featureNum = jTensor.shape(1)
    jTensor.storage.grouped(featureNum).map(array => {
      Tensor[Float](array, Array(array.length))
    }).toArray
  }
  def fgBoostFit(model: FGBoostModel, feature: JTensor, label: JTensor, boostRound: Int) = {
    val tensorArray = jTensorToTensorArray(feature)
    val labelArray = if (label != null) label.storage else null
    model.fit(tensorArray, labelArray, boostRound)
  }
  def fgBoostEvaluate(model: FGBoostModel, feature: JTensor, label: JTensor) = {
    val tensorArray = jTensorToTensorArray(feature)
    val labelArray = if (label != null) label.storage else null
    model.evaluate(tensorArray, labelArray)
  }
  def fgBoostPredict(model: FGBoostModel, feature: JTensor) = {
    val tensorArray = jTensorToTensorArray(feature)
    val result = model.predict(tensorArray).map(_.storage().array())
    JTensor(result.flatten, Array(result.length, result(0).length), bigdlType = "float")
  }
  def nnFit(model: FLModel) = {

  }
  def nnEvaluate(model: FLModel) = {

  }
  def nnPredict(model: FLModel) = {

  }

  // Pytorch support
  def pytorchTrainStep(pred: JTensor, target: JTensor, version: Int, algorithm: String) = {
    PytorchSuite.trainStep(pred, target, version, algorithm)
  }
  def pytorchEvaluateStep() = {
    // TODO
  }
  def pytorchPredictStep() = {
    // TODO
  }
}
