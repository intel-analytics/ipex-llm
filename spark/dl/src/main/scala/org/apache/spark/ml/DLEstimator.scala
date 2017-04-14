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
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.example.imageclassification.MlUtils._
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 * A wrapper of Optimizer to support fit() in ML Pipelines as an Estimator
 *
 * @param module model to be optimized
 * @param criterion criterion to be used
 * @param batchShape batch shape to be used
 */

class DLEstimator[T : ClassTag]
  (module : Module[T], criterion : Criterion[T], batchShape : Array[Int])
  (override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends MLEstimator{

  private def validateParameters(): Unit = {

    require(null != module,
      "DLEstimator: module must not be null")

    require(null != criterion,
      "DLEstimator: criterion must not be null")

  }


  override def process(dataFrame: DataFrame): MlTransformer = {

    this.validateParameters()

    val rdd : RDD[MiniBatch[T]] = toMinibatch(dataFrame)

    val dataset = DataSet.rdd(rdd)

    val optimizer = Optimizer(module, dataset, criterion)

    val estimatedModule = optimizer.optimize()

    var classifier = new DLClassifier[T]()
      .setInputCol("features")
      .setOutputCol("predict")

    val paramsTrans = ParamMap(
      classifier.modelTrain -> estimatedModule,
      classifier.batchShape -> batchShape)

    classifier = classifier.copy(paramsTrans)

    classifier
  }

  private def toMinibatch(df : DataFrame) : RDD[MiniBatch[T]] = {

    val sampleDF = df.select("minibatch")

    val dfRows : RDD[Row] = sampleDF.rdd

    val minibatchRDD : RDD[MiniBatch[T]] = dfRows.map(row => {
      val columnData = row.get(0).asInstanceOf[GenericRowWithSchema]

      val featureData = columnData.get(0).asInstanceOf[mutable.WrappedArray[T]].toArray
      val featureSize = columnData.get(1).asInstanceOf[mutable.WrappedArray[Int]].toArray

      val featureStorage = Storage(featureData)

      val labelData = columnData.get(2).asInstanceOf[mutable.WrappedArray[T]].toArray
      val labelSize = columnData.get(3).asInstanceOf[mutable.WrappedArray[Int]].toArray

      val labelStorage = Storage(labelData)

      val featureTensor = Tensor(featureStorage, 1, featureSize)

      val labelTensor = Tensor(labelStorage, 1, labelSize)

      val miniBatch : MiniBatch[T] = MiniBatch(featureTensor, labelTensor)
      miniBatch

    })
    minibatchRDD
  }
}

case class DLEstimatorData[T](data : DLEstimatorMinibatchData[T])

case class DLEstimatorMinibatchData[T](featureData : Array[T], featureSize : Array[Int],
                            labelData : Array[T], labelSize : Array[Int])



