package com.intel.analytics.bigdl.ppml.python

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.python.api.PythonBigDL
import com.intel.analytics.bigdl.ppml.{FLContext, FLModel, FLServer}
import com.intel.analytics.bigdl.ppml.fgboost.FGBoostModel

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
  def createFGBoostRegression() = {

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
  def fgBoostFit(model: FGBoostModel) = {

  }
  def fgBoostEvaluate(model: FGBoostModel) = {

  }
  def fgBoostPredict(model: FGBoostModel) = {

  }
  def nnFit(model: FLModel) = {

  }
  def nnEvaluate(model: FLModel) = {

  }
  def nnPredict(model: FLModel) = {

  }
}
