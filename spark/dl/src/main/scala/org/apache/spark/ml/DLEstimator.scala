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
package org.apache.spark.ml
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch}
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasInputCols, HasLabelCol}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * A wrapper of Optimizer to support fit() in ML Pipelines as an Estimator
 * feature column name and label column name should be provided in training Dataframe
 * Model to be trained, Feature size, label size, batch shape size must also be provided
 * For details usage, please refer to example :
 * [[com.intel.analytics.bigdl.example.MLPipeline.DLEstimatorLeNet]]
 */
class DLEstimator[@specialized(Float, Double) T: ClassTag]
(override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends MLEstimator with HasFeaturesCol with HasLabelCol with DLDataParams[T] {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def validateParameters(): Unit = {
    val params = this.extractParamMap()
    require(isDefined(modelTrain),
      "DLEstimator: model for optimization must not be null")
    require(isDefined(batchShape),
      "DLEstimator: batchShape for optimization must not be null")
    require(isDefined(featuresCol),
      "DLEstimator: features data must not be null")
    require(isDefined(featureSize),
      "DLEstimator: features size col must not be null")
    require(isDefined(labelCol),
      "DLEstimator: label data must not be null")
    require(isDefined(labelSize),
      "DLEstimator: label size must not be null")
    require(isDefined(criterion),
      "DLEstimator: criterion must not be null")
  }

  override def process(dataFrame: DataFrame): MlTransformer = {

    validateParameters()

    val batches = toMiniBatch(dataFrame)

    val dataset = DataSet.rdd(batches)

    val optimizer = Optimizer($(modelTrain), dataset, $(criterion))

    var optimizedModule = $(modelTrain)

    optimizedModule = optimizer.optimize()

    var classifier = new DLClassifier[T]()
      .setInputCol("features")
      .setOutputCol("predict")

    val paramsTrans = ParamMap(
      classifier.modelTrain -> optimizedModule,
      classifier.batchShape -> ${batchShape})

    classifier = classifier.copy(paramsTrans)

    classifier
  }

  private def toMiniBatch(dataFrame: DataFrame) : RDD[MiniBatch[T]] = {

    val data = dataFrame.rdd
    val batchs = data.map(row => {
      val featureData = row.getAs[mutable.WrappedArray[T]]($(featuresCol)).toArray
      val labelData = row.getAs[mutable.WrappedArray[T]]($(labelCol)).toArray
      MiniBatch[T](Tensor(featureData, $(featureSize)), Tensor(labelData, $(labelSize)))
    })
    batchs
  }

  override def copy(extra: ParamMap): DLEstimator[T] = {
    copyValues(new DLEstimator(uid), extra)
  }
}

private[ml] trait DLDataParams[@specialized(Float, Double) T] extends Params {

  final val modelTrain = new Param[Module[T]](this, "module factory", "network model")

  final val criterion = new Param[Criterion[T]](this, "criterion factory", "criterion for optimize")

  final val featureSize = new Param[Array[Int]](this, "feature size", "feature input size")

  final val labelSize = new Param[Array[Int]](this, "label size", "label input size")

  final val batchShape = new Param[Array[Int]](this, "batch size", "batch size for input")

  final def getModel: Module[T] = $(modelTrain)

  final def getCriterion : Criterion[T] = $(criterion)

  final def getFeatureSize : Array[Int] = $(featureSize)

  final def getLabelSize : Array[Int] = $(labelSize)

  final def getBatchShape : Array[Int] = $(batchShape)

}



