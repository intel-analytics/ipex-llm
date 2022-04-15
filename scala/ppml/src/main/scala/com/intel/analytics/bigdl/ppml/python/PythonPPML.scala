package com.intel.analytics.bigdl.ppml.python


import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.python.api.{JTensor, PythonBigDL}
import com.intel.analytics.bigdl.ppml.algorithms.{FGBoostRegression, PSI}
import com.intel.analytics.bigdl.ppml.{FLClient, FLContext, FLServer, NNModel}
import com.intel.analytics.bigdl.ppml.fgboost.FGBoostModel
import com.intel.analytics.bigdl.ppml.utils.{FLClientClosable, TimingSupportive}

import java.util.{List => JList}
import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dllib.utils.Log4Error

object PythonPPML {

  def ofFloat(): PythonPPML[Float] = new PythonPPML[Float]()

  def ofDouble(): PythonPPML[Double] = new PythonPPML[Double]()
}
class PythonPPML[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL with TimingSupportive{
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
  def flServerBlockUntilShutdown(flServer: FLServer) = {
    flServer.blockUntilShutdown()
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
    Log4Error.invalidOperationError(jTensor.shape.length == 2,
      s"FGBoost only support 2D input, get dimension: ${jTensor.shape.length}")
    val featureNum = jTensor.shape(1)
    jTensor.storage.grouped(featureNum).map(array => {
      Tensor[Float](array, Array(array.length))
    }).toArray
  }
  def fgBoostFitAdd(model: FGBoostModel, xTrain: JTensor) = {
    val tensorArray = timing("JVM JTensor to Array Tensor") {
      jTensorToTensorArray(xTrain)
    }
    timing("Add training batch to model") {
      model.fitAdd(tensorArray)
    }


  }
  def fgBoostFitCall(model: FGBoostModel, yTrain: JTensor, boostRound: Int) = {
    val labelArray = if (yTrain != null) yTrain.storage else null
    timing("Call fit method") {
      model.fitCall(labelArray, boostRound)
    }

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
  def nnFit(model: NNModel) = {

  }
  def nnEvaluate(model: NNModel) = {

  }
  def nnPredict(model: NNModel) = {

  }
}
