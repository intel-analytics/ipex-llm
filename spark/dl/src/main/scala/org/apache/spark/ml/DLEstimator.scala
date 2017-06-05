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
import org.apache.spark.sql.types.{ArrayType, StructType}

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * A wrapper of Optimizer to support fit() in ML Pipelines as an Estimator
 * feature column name and label column name should be provided in training Dataframe
 * Model to be trained, Feature size, label size, batch shap must also be provided
 * For details usage, please refer to example :
 * [[com.intel.analytics.bigdl.example.MLPipeline.DLEstimatorLeNet]]
 *
 * @param modelTrain module to be optimized
 * @param criterion  criterion method
 * @param batchShape batch shape for DLClassifier transformation input
 */
class DLEstimator[@specialized(Float, Double) T: ClassTag]
(val modelTrain : Module[T], val criterion : Criterion[T], val batchShape : Array[Int],
 override val uid: String = "DLEstimator")
(implicit ev: TensorNumeric[T])
  extends DLEstimatorBase with HasFeaturesCol with HasLabelCol with DLDataParams[T] {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  private def validateInput(schema : StructType): Unit = {
    require(isDefined(featuresCol),
      "DLEstimator: features data must not be null")
    require(isDefined(featureSize),
      "DLEstimator: features size col must not be null")
    require(isDefined(labelCol),
      "DLEstimator: label data must not be null")
    require(isDefined(labelSize),
      "DLEstimator: label size must not be null")
    val featureIndex = schema.fieldIndex($(featuresCol))
    val featureField = schema.fields(featureIndex)
    require(featureField.dataType.isInstanceOf[ArrayType], "Feature data should be of array type")
    val labelIndex = schema.fieldIndex($(labelCol))
    val labelField = schema.fields(labelIndex)
    require(labelField.dataType.isInstanceOf[ArrayType], "Label data should be of array type")
  }

  override protected def process(dataFrame: DataFrame): DLTransformer = {

    validateInput(dataFrame.schema)

    val batches = toMiniBatch(dataFrame)

    val dataset = DataSet.rdd(batches)

    val optimizer = Optimizer(modelTrain, dataset, criterion)

    var optimizedModule = modelTrain

    optimizedModule = optimizer.optimize()

    var classifier = new DLClassifier[T]()

    val paramsTrans = ParamMap(
      classifier.modelTrain -> optimizedModule,
      classifier.batchShape -> batchShape)

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
    copyValues(new DLEstimator(modelTrain, criterion, batchShape), extra)
  }
}

private[ml] trait DLDataParams[@specialized(Float, Double) T] extends Params {

  final val featureSize = new Param[Array[Int]](this, "feature size", "feature input size")

  final val labelSize = new Param[Array[Int]](this, "label size", "label input size")

  final def getFeatureSize : Array[Int] = $(featureSize)

  final def getLabelSize : Array[Int] = $(labelSize)

}



