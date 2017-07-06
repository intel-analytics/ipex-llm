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
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample, SampleToBatch}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, MklBlas}


import scala.reflect.ClassTag

object LocalPredictor {
  def getAndClearWeightBias[T: ClassTag](parameters: (Array[Tensor[T]], Array[Tensor[T]]))
                                        (implicit ev: TensorNumeric[T]): Array[Tensor[T]] = {
    var i = 0
    val weightsBias = new Array[Tensor[T]](parameters._1.length)
    while (i < parameters._1.length) {
      if (parameters._1(i) != null) {
        val wb = parameters._1(i)
        weightsBias(i) = Tensor[T](Storage(wb.storage().array()),
          wb.storageOffset(), wb.size(), wb.stride())
      }
      i += 1
    }
    i = 0
    while (i < parameters._1.length) {
      if (parameters._1(i) != null) {
        parameters._1(i).set()
      }
      if (parameters._2(i) != null) {
        parameters._2(i).set()
      }
      i += 1
    }
    weightsBias
  }

  def apply[T: ClassTag](model: Module[T], weightsBias: Array[Tensor[T]])
                        (implicit ev: TensorNumeric[T]): LocalPredictor[T] = {
    new LocalPredictor[T](model, weightsBias)
  }
}

class LocalPredictor[T: ClassTag] private[optim](model: Module[T], weightsBias: Array[Tensor[T]])
                                                (implicit ev: TensorNumeric[T])
  extends Serializable {

  val logger = LocalValidator.logger
  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case _ => throw new IllegalArgumentException
  }

  private val batchSize = 8

  def predictClass(dataSet: Array[Sample[T]]): Array[Int] = {
    val result = predict(dataSet)
    result.map(output => {
      val _output = output.toTensor[T]
      require(_output.dim() == 1, s"Predictor.predictClass:" +
        s"Only support one sample has one lable, but got ${_output.dim()} label")
      ev.toType[Int](_output.max(1)._2.valueAt(1))
    })
  }

  def predictClass(dataSet: LocalDataSet[Sample[T]]): Array[Int] = {
    val result = predict(dataSet)
    result.map(output => {
      val _output = output.toTensor[T]
      require(_output.dim() == 1, s"Predictor.predictClass:" +
        s"Only support one sample has one lable, but got ${_output.dim()} label")
      ev.toType[Int](_output.max(1)._2.valueAt(1))
    })
  }

  def predict(dataSet: LocalDataSet[Sample[T]]): Array[Activity] = {
    val dataset = dataSet.transform(SampleToBatch[T](
      batchSize = batchSize, None, None, None,
      partitionNum = Some(1))).asInstanceOf[LocalDataSet[MiniBatch[T]]]
    val dataIter = dataset.data(train = false)

    val workingModels = (1 to subModelNumber).map(_ => {
      val submodel = model.cloneModule().evaluate()
      putWeightBias(weightsBias, submodel)
      submodel
    }).toArray

    dataIter.map(batch => {
      println("Enter map")
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val start = System.nanoTime()
      val result = Engine.default.invokeAndWait(
        (0 until parallelism).map(b =>
          () => {
            val offset = b * stackSize + math.min(b, extraSize) + 1
            val length = stackSize + (if (b < extraSize) 1 else 0)
            val currentMiniBatch = batch.slice(offset, length)
            val input = currentMiniBatch.getInput()
            val output = workingModels(b).forward(input).toTensor[T]
            output
          }
        )
      )
      val batchResult = result.flatMap(_.split(1)).map(_.asInstanceOf[Activity])
      batchResult
    }).toArray.flatten

  }

  def predict(dataSet: Array[Sample[T]]): Array[Activity] = {
    val iter = dataSet.iterator
    val transformer = SampleToBatch[T](
      batchSize = batchSize, None, None, None,
      partitionNum = Some(1))
    val dataIter = transformer(iter)

    val workingModels = (1 to subModelNumber).map(_ => {
      val submodel = model.cloneModule().evaluate()
      putWeightBias(weightsBias, submodel)
      submodel
    }).toArray

    dataIter.map(batch => {
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val start = System.nanoTime()
      val result = Engine.default.invokeAndWait(
        (0 until parallelism).map(b =>
          () => {
            val offset = b * stackSize + math.min(b, extraSize) + 1
            val length = stackSize + (if (b < extraSize) 1 else 0)
            val currentMiniBatch = batch.slice(offset, length)
            val input = currentMiniBatch.getInput()
            val output = workingModels(b).forward(input).toTensor[T]
            output

          }
        )
      )
      val batchResult = result.flatMap(_.split(1)).map(_.asInstanceOf[Activity])
      batchResult
    }).toArray.flatten

  }

  private def putWeightBias(weightBias: Array[Tensor[T]],
                            localModel: Module[T]): Unit = {
    val localWeightBias = localModel.parameters()._1
    var i = 0
    while (i < localWeightBias.length) {
      if (localWeightBias(i) != null) {
        localWeightBias(i).set(weightBias(i))
      }
      i += 1
    }
  }

}


