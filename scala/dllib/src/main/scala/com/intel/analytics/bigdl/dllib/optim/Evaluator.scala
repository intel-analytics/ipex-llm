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
import com.intel.analytics.bigdl.dataset.{Sample, SampleToBatch}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
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

    val modelBroad = ModelBroadcast[T].broadcast(dataset.sparkContext, model.evaluate())
    val partitionNum = dataset.partitions.length

    val totalBatch = batchSize.getOrElse(batchPerPartition * partitionNum)
    val otherBroad = dataset.sparkContext.broadcast(vMethods, SampleToBatch(
      batchSize = totalBatch, None, None, None, partitionNum = Some(partitionNum)))

    dataset.mapPartitions(partition => {
      val localModel = modelBroad.value()
      val localMethod = otherBroad.value._1.map(_.clone())
      val localTransformer = otherBroad.value._2.cloneTransformer()
      val miniBatch = localTransformer(partition)
      miniBatch.map(batch => {
        val output = localModel.forward(batch.data)
        localMethod.map(validation => {
          validation(output, batch.labels)
        })
      })
    }).reduce((left, right) => {
        left.zip(right).map { case (l, r) => l + r }
    }).zip(vMethods)
  }
}
