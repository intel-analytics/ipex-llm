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
package com.intel.analytics.bigdl.utils

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.optim.LocalPredictor

import scala.reflect.ClassTag


object LocalModule {
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

  def apply[T: ClassTag](model: Module[T])
                        (implicit ev: TensorNumeric[T]): LocalModule[T] = {
    val weightsBias = getAndClearWeightBias(model.parameters())
    new LocalModule[T](model, weightsBias)
  }
}

class LocalModule[T: ClassTag] private(model: Module[T], weightsBias: Array[Tensor[T]])
                                                (implicit ev: TensorNumeric[T])
  extends Serializable {

  private val predictor = LocalPredictor(model, weightsBias)

  def predictClass(dataSet: Array[Sample[T]]): Array[Int] = {
    predictor.predictClass(dataSet)
  }

  def predictClass(dataSet: LocalDataSet[MiniBatch[T]]): Array[Int] = {
    predictor.predictClass(dataSet)
  }

  def predict(dataSet: LocalDataSet[MiniBatch[T]]): Array[Activity] = {
    predictor.predict(dataSet)
  }

  def predict(dataSet: Array[Sample[T]]): Array[Activity] = {
    predictor.predict(dataSet)
  }
}

