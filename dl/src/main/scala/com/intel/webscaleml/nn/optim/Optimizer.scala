package com.intel.webscaleml.nn.optim

import com.intel.webscaleml.nn.nn.{Criterion, Module}
import com.intel.webscaleml.nn.tensor.{T, Table, Tensor, torch}
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

import scala.util.Random

/**
  * Train a neural network model on a distributed data set
  *
  * @param module module to be optimized
  * @param criterion cost function
  * @param dataSet distributed data set
  * @tparam T numeric type of model
  */
abstract class Optimizer[@specialized(Float, Double) T](val module : Module[T], val criterion : Criterion[T],
    dataSet: DataSet[_, T]) extends Serializable with Logging
  with HasCrossValidation[T] with ModelPersist[T] {

  import Optimizer._

  def optimize() : Module[T]

  // We pre-create models on each partition of the data set
  private def init() = {
    val broadcast = dataSet.getSparkContext().broadcast((module, criterion))
    val models = dataSet.partitions().mapPartitions(_ => {
      val (broadcastModule, broadcastCriterion) = broadcast.value
      val localModule = broadcastModule.cloneModule()
      val localCriterion = broadcastCriterion.cloneCriterion()
      val (weights, grads) = localModule.getParameters()
      Iterator.single(CachedModel(localModule, localCriterion, weights, grads, T()))
    }).persist()
    models.setName("modelRDD")
    logInfo("Cache models...")
    models.count()
    logInfo("Cache models... done")
    models
  }

  val models = init()
}

object Optimizer {

  /**
    * Represent a cached module and its cost function
    * @param model module instance
    * @param criterion cost function instance
    * @param weight a single tensor storing all parameters of the module
    * @param gradient a single tensor storing all gradient of the parameters of the module
    * @param state contains train state
    * @tparam T
    */
  case class CachedModel[T](model : Module[T], criterion: Criterion[T], weight : Tensor[T], gradient : Tensor[T], state : Table)
}
