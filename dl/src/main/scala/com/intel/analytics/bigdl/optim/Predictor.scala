/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{Sample, SampleToBatch, DataSet => _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

abstract class Predictor[T, D](model: Module[T], dataSet: RDD[D]) extends Serializable {
  def predict(batchSize: Int): RDD[Tensor[T]]
}

object Predictor {
  def apply[T: ClassTag, D](model: Module[T], dataset: RDD[D])
   (implicit ev: TensorNumeric[T]): Predictor[T, D] = {
    dataset match {
      case d: RDD[Sample[_]] =>
        new DistriPredictor[T](
          model = model,
          dataSet = dataset.asInstanceOf[RDD[Sample[T]]]
        ).asInstanceOf[Predictor[T, D]]
      case _ =>
        throw new UnsupportedOperationException
    }

  }
}

class DistriPredictor[T: ClassTag] private[optim](
   model: Module[T],
   dataSet: RDD[Sample[T]]
 )(implicit ev: TensorNumeric[T]) extends Predictor[T, Sample[T]](model, dataSet) {

  def predict(batchSize: Int) : RDD[Tensor[T]] = {
    val modelBroadCast = dataSet.sparkContext.broadcast(
      model.evaluate(), new SampleToBatch[T](batchSize))
    dataSet.mapPartitions { partition =>
      val localModel = modelBroadCast.value._1.cloneModule()
      val localTransformer = modelBroadCast.value._2.cloneTransformer()
      val miniBatch = localTransformer(partition)
      miniBatch.map { batch =>
        val output = localModel.forward(batch.data).toTensor[T]
        output
      }
    }
  }
}
