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

import com.intel.analytics.bigdl.dataset.{DataSet => DataSource, Batch}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, MklDnn}
import org.apache.spark.rdd.RDD

class DistriValidator[T](
  model: Module[T]
) extends Validator[T, RDD[Batch[T]]](model) {

  override def test(
    dataSet: DataSource[RDD[Batch[T]]],
    vMethods: Array[ValidationMethod[T]])
  : Array[(ValidationResult, ValidationMethod[T])] = {
    val rdd = dataSet.data(looped = false)
    val broadcastModel = rdd.sparkContext.broadcast(model.evaluate())

    rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value.cloneModule().evaluate()
      dataIter.map(batch => {
        require(batch.data.size(1) == batch.labels.size(1))
        val input = batch.data
        val target = batch.labels
        val output = localModel.forward(input)
        vMethods.map(validation => {
          validation(output, target)
        })
      })
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods)
  }
}
