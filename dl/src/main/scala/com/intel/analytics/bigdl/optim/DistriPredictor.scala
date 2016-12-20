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
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, MklBlas, MklDnn}
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag

object DistriPredictor {
  val logger = Logger.getLogger(this.getClass)
}

class DistriPredictor[T: ClassTag](model: Module[T])(implicit ev: TensorNumeric[T])
  extends Predictor[T, RDD[Tensor[T]], RDD[Tensor[T]]]{

  import DistriPredictor._

  override def predict(dataSet: dataset.DataSet[RDD[Tensor[T]]])
  : dataset.DataSet[RDD[Tensor[T]]] = {
    val rdd = dataSet.data(looped = false)
    val broadcastModel = rdd.sparkContext.broadcast(model.evaluate())
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case MklDnn => 1
    }
    val _ev = ev
    val result = rdd.mapPartitions(dataIter => {
      val localModel = broadcastModel.value
      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      val workingModels = (1 to _subModelNumber)
        .map(_ => localModel.cloneModule().evaluate()).toArray
      dataIter.map(batch => {
        val stackSize = batch.size(1) / _subModelNumber
        val extraSize = batch.size(1) % _subModelNumber
        val parallelism = if (stackSize == 0) extraSize else _subModelNumber
        val result = Engine.default.invokeAndWait(
          (0 until parallelism).map(b =>
            () => {
              val offset = b * stackSize + math.min(b, extraSize)
              val length = stackSize + (if (b < extraSize) 1 else 0)
              val input = batch.narrow(1, offset + 1, length)
              workingModels(b).forward(input).toTensor[T](_ev)
            }
          )
        )
        val resultSize = result.head.size()
        resultSize(0) = batch.size(1)
        val resultTensor = Tensor[T](resultSize)
        Engine.default.invokeAndWait(
          (0 until parallelism).map(b =>
            () => {
              val offset = b * stackSize + math.min(b, extraSize)
              val length = stackSize + (if (b < extraSize) 1 else 0)
              resultTensor.narrow(1, offset + 1, length).copy(result(b))
            }
          )
        )
        resultTensor
      })
    })

    new dataset.DataSet[RDD[Tensor[T]]] {
      override def data(looped: Boolean): RDD[Tensor[T]] = result
      override def size(): Long = dataSet.size()
      override def shuffle(): Unit = dataSet.shuffle()
    }
  }
}
