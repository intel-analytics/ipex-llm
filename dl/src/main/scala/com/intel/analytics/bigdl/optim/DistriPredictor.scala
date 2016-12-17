/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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
import com.intel.analytics.bigdl.dataset.DistributedDataSet
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.rdd.RDD

class DistriPredictor[T](model: Module[T])(implicit ev: TensorNumeric[T])
  extends Predictor[T, RDD[Tensor[T]], RDD[Tensor[T]]]{

  override def predict(dataSet: dataset.DataSet[RDD[Tensor[T]]])
  : dataset.DataSet[RDD[Tensor[T]]] = {
    val rdd = dataSet.data(looped = false)
    val broadcastModel = rdd.sparkContext.broadcast(model.evaluate())

    val result = rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value.cloneModule().evaluate()
      dataIter.map(batch => {
        val input = batch
        localModel.forward(input).toTensor[T]
      })
    })

    new dataset.DataSet[RDD[Tensor[T]]] {
      override def data(looped: Boolean): RDD[Tensor[T]] = result
      override def size(): Long = dataSet.size()
      override def shuffle(): Unit = dataSet.shuffle()
    }
  }
}
