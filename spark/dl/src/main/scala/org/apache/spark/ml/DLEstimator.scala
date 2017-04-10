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
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

/**
 * A wrapper of Optimizer to support fit() in ML Pipelines as an Estimator
 *
 * @param module model to be optimized
 * @param criterion criterion to be used
 * @param batchShape batch shape to be used
 */

class DLEstimator[T : ClassTag]
  (module : Module[T], criterion : Criterion[T], batchShape : Array[Int], batchSize: Int)
  (override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends MLEstimator{

  private def validateParameters(): Unit = {

    require(null != module,
      "DLEstimator: module for estimator must not be null")

    require(null != criterion,
      "DLEstimator: criterion must not be null")

  }


  override def process(dataFrame: DataFrame): MlTransformer = {

    this.validateParameters()

    val rdd : RDD[Sample[T]] = toSample(dataFrame)

    val optimizer = Optimizer(module, rdd, criterion, batchSize)

    val estimatedModule = optimizer.optimize()

    val classifier = new DLClassifier[T]()
      .setInputCol("features")
      .setOutputCol("predict")

    classifier.modelTrain -> estimatedModule

    classifier.batchShape -> batchShape

    classifier
  }

  private def toSample(df : DataFrame) : RDD[Sample[T]] = {

    val dfRows : RDD[Row] = df.rdd

    val sampleRDD : RDD[Sample[T]] = dfRows.map(row => {

      val featureData = row.get(0).asInstanceOf[Seq[T]].toArray
      val featureSize = row.get(1).asInstanceOf[Seq[Int]].toArray
      val featureStride = row.get(2).asInstanceOf[Seq[Int]].toArray

      val featureStorage = Storage(featureData)

      val labelData = row.get(3).asInstanceOf[Seq[T]].toArray
      val labelSize = row.get(4).asInstanceOf[Seq[Int]].toArray
      val labelStride = row.get(5).asInstanceOf[Seq[Int]].toArray

      val labelStorage = Storage(labelData)

      val featureTensor = Tensor(featureStorage, 1, featureSize, featureStride)

      val labelTensor = Tensor(labelStorage, 1, labelSize, labelStride)

      val sample : Sample[T] = Sample(featureTensor, labelTensor)

      sample

    })
    sampleRDD
  }
}
case class DLEstimatorData[T](featureData : Array[T], featureSize : Array[Int],
                            featureStride : Array[Int], labelData : Array[T],
                            labelSize : Array[Int], labelStrude : Array[Int])



