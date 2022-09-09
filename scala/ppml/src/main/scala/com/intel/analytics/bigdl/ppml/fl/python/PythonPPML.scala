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

package com.intel.analytics.bigdl.ppml.fl.python


import com.intel.analytics.bigdl.dllib.optim.ValidationResult
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.ppml.fl.algorithms.{FGBoostRegression, PSI}
import com.intel.analytics.bigdl.ppml.fl.fgboost.FGBoostModel
import com.intel.analytics.bigdl.ppml.fl.utils.{FLClientClosable, TimingSupportive}

import java.util.{List => JList}
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.fl.{FLClient, FLContext, FLServer, NNModel}

object PythonPPML {

  def ofFloat(): PythonPPML[Float] = new PythonPPML[Float]()

  def ofDouble(): PythonPPML[Double] = new PythonPPML[Double]()
}
class PythonPPML[T: ClassTag](implicit ev: TensorNumeric[T])
  extends PythonBigDL with TimingSupportive {
  def initFLContext(id: Int, target: String): Unit = {
    FLContext.initFLContext(id, target)
  }
  def setPsiSalt(psiSalt: String): Unit = {
    FLContext.setPsiSalt(psiSalt)
  }
  def createFLServer(): FLServer = {
    new FLServer()
  }
  def createHflNN(): Unit = {

  }
  def createHflLogisticRegression(): Unit = {

  }
  def createHflLinearRegression(): Unit = {

  }
  def createVflLogisticRegression(): Unit = {

  }
  def createVflLinearRegression(): Unit = {

  }

  def createFGBoostRegression(learningRate: Double,
                              maxDepth: Int,
                              minChildSize: Int,
                              serverModelPath: String): FGBoostRegression = {
    new FGBoostRegression(learningRate.toFloat, maxDepth, minChildSize, serverModelPath)
  }
  def fgBoostLoadServerModel(fgBoost: FGBoostModel, modelPath: String): Unit = {
    fgBoost.loadServerModel(modelPath)
  }
  def createFGBoostClassification(): Unit = {

  }
  def flServerBuild(flServer: FLServer): Unit = {
    flServer.build()
  }
  def flServerStart(flServer: FLServer): Unit = {
    flServer.start()
  }
  def flServerStop(flServer: FLServer): Unit = {
    flServer.stop()
  }

  def flServerSetClientNum(flServer: FLServer, clientNum: Int): Unit = {
    flServer.setClientNum(clientNum)
  }
  def flServerSetPort(flServer: FLServer, port: Int): Unit = {
    flServer.setPort(port)
  }
  def flServerBlockUntilShutdown(flServer: FLServer): Unit = {
    flServer.blockUntilShutdown()
  }

  /**
   * FlClient is not exposed to users API, the Python API for this only locates in test
   * @param target the FlClient target Url
   * @return
   */
  def createFLClient(target: String): FLClient = {
    val flClient = new FLClient()
    if (target != null) flClient.setTarget(target)
    flClient
  }
  def flClientClosableSetFLClient(flClientClosable: FLClientClosable,
                                  flClient: FLClient): FLClientClosable = {
    flClientClosable.setFlClient(flClient)
  }
  def createPSI(): PSI = {
    new PSI()
  }
  def psiGetSalt(psi: PSI, secureCode: String = ""): String = {
    psi.getSalt(secureCode)
  }
  def psiUploadSet(psi: PSI, ids: JList[String], salt: String): Unit = {
    psi.uploadSet(ids, salt)
  }
  def psiDownloadIntersection(psi: PSI,
                              maxtry: Int = 100, retry: Int = 3000): java.util.List[String] = {
    psi.downloadIntersection(maxtry, retry)
  }
  def psiGetIntersection(psi: PSI, ids: JList[String],
                         maxtry: Int = 100, retry: Int = 3000): java.util.List[String] = {
    psi.getIntersection(ids, maxtry, retry)
  }
  def jTensorToTensorArray(jTensor: JTensor): Array[Tensor[Float]] = {
    Log4Error.invalidOperationError(jTensor.shape.length == 2,
      s"FGBoost only support 2D input, get dimension: ${jTensor.shape.length}")
    val featureNum = jTensor.shape(1)
    jTensor.storage.grouped(featureNum).map(array => {
      Tensor[Float](array, Array(array.length))
    }).toArray
  }
  def fgBoostFitAdd(model: FGBoostModel, xTrain: JTensor): ArrayBuffer[Tensor[Float]] = {
    val tensorArray = timing("JVM JTensor to Array Tensor") {
      jTensorToTensorArray(xTrain)
    }
    timing("Add training batch to model") {
      model.fitAdd(tensorArray)
    }


  }
  def fgBoostFitCall(model: FGBoostModel, yTrain: JTensor, boostRound: Int): Unit = {
    logger.info(s"start call fit")
    val labelArray = if (yTrain != null) yTrain.storage else null
    timing("Call fit method") {
      model.fitCall(labelArray, boostRound)
    }

  }

  def fgBoostFit(model: FGBoostModel, feature: JTensor, label: JTensor, boostRound: Int): Unit = {
    val tensorArray = jTensorToTensorArray(feature)
    val labelArray = if (label != null) label.storage else null
    model.fit(tensorArray, labelArray, boostRound)
  }
  def fgBoostEvaluate(model: FGBoostModel,
                      feature: JTensor, label: JTensor): Array[ValidationResult] = {
    val tensorArray = jTensorToTensorArray(feature)
    val labelArray = if (label != null) label.storage else null
    model.evaluate(tensorArray, labelArray)
  }
  def fgBoostPredict(model: FGBoostModel, feature: JTensor): JTensor = {
    val tensorArray = jTensorToTensorArray(feature)
    val result = model.predict(tensorArray).map(_.storage().array())
    JTensor(result.flatten, Array(result.length, result(0).length), bigdlType = "float")
  }
  def fgBoostRegressionSave(model: FGBoostRegression, dest: String): Unit = {
    model.saveModel(dest)
  }
  def fgBoostRegressionLoad(src: String): FGBoostRegression = {
    FGBoostRegression.loadModel(src)
  }
  def nnFit(model: NNModel): Unit = {

  }
  def nnEvaluate(model: NNModel): Unit = {

  }
  def nnPredict(model: NNModel): Unit = {

  }
}
