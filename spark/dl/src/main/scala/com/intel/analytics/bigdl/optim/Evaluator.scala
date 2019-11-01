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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, MklDnn}
import com.intel.analytics.bigdl.utils.intermediate.ConversionUtils
import org.apache.spark.rdd
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object Evaluator {
  def apply[T: ClassTag](model: Module[T])(implicit ev: TensorNumeric[T]): Evaluator[T] = {
    new Evaluator[T](model)
  }
}

/**
 * model evaluator
 * @param model model to be evaluated
 */
class Evaluator[T: ClassTag] private[optim](model: Module[T])(implicit ev: TensorNumeric[T])
  extends Serializable {

  private val batchPerPartition = 4

  /**
   * Applies ValidationMethod to the model and rdd dataset.
   * @param vMethods
   * @param batchSize total batchsize
   * @return
   */
  def test(dataset: RDD[Sample[T]],
   vMethods: Array[ValidationMethod[T]],
   batchSize: Option[Int] = None): Array[(ValidationResult, ValidationMethod[T])] = {

    val partitionNum = dataset.partitions.length
    val totalBatch = batchSize.getOrElse(batchPerPartition * partitionNum)

    val dummyInput = Predictor.getDummyData(dataset, totalBatch / partitionNum)

    val modelBroad = ModelBroadcast[T]().broadcast(dataset.sparkContext,
      ConversionUtils.convert(model.evaluate()), dummyInput)
    val rdd = ConversionUtils.coalesce(dataset)
    val otherBroad = rdd.sparkContext.broadcast(vMethods, SampleToMiniBatch(
      batchSize = totalBatch, partitionNum = Some(rdd.partitions.length)))

    rdd.mapPartitions(partition => {
      val localModel = modelBroad.value(false, true, dummyInput)
      val localMethod = otherBroad.value._1.map(_.clone())
      val localTransformer = otherBroad.value._2.cloneTransformer()
      val miniBatch = localTransformer(partition)
      miniBatch.map(batch => {
        val output = localModel.forward(batch.getInput())
        localMethod.map(validation => {
          validation(output, batch.getTarget())
        })
      })
    }).reduce((left, right) => {
        left.zip(right).map { case (l, r) => l + r }
    }).zip(vMethods)
  }

  /**
   * Apply ValidationMethod to the model and rdd dataset.
   * @param vMethods
   * @return
   */
  private[bigdl] def testMiniBatch(dataset: RDD[MiniBatch[T]],
           vMethods: Array[ValidationMethod[T]]
          ): Array[(ValidationResult, ValidationMethod[T])] = {

    val rdd = ConversionUtils.coalesce(dataset)
    val modelBroad = ModelBroadcast[T]().broadcast(rdd.sparkContext,
      ConversionUtils.convert(model.evaluate()))
    val otherBroad = rdd.sparkContext.broadcast(vMethods)


    rdd.mapPartitions(miniBatch => {
      val localModel = modelBroad.value()
      val localMethod = otherBroad.value
      miniBatch.map(batch => {
        val output = localModel.forward(batch.getInput())
        localMethod.map(validation => {
          validation(output, batch.getTarget())
        })
      })
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) => l + r }
    }).zip(vMethods)
  }

}
