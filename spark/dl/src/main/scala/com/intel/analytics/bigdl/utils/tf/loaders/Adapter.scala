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
package com.intel.analytics.bigdl.utils.tf.loaders

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Sometimes parameters pass as tensor from some previous nodes in graph. So we can't construct
 * module at module define time. In such case, you can use the AdapterOperation to wrapper the
 * module and create it in runtime.
 *
 * Please note you must guarantee the input parameter won't change each time.
 * @param configIndexes configuration tensor indexes, start from 1 and -1 specify the last one
 * @tparam T Numeric type. Only support float/double now
 */
abstract class Adapter[T: ClassTag](
  val configIndexes: Array[Int]
)(implicit ev: TensorNumeric[T])
  extends AbstractModule[Table, Activity, T]{

  private var module : AbstractModule[Activity, Activity, T] = _
  private var indexes : Array[Int] = _
  private var dataIndexes: Array[Int] = _
  private var zeroGrads: Array[Tensor[_]] = _
  private var realInput: Activity = _
  private var initTensors: Array[Tensor[_]] = _

  def build(tensorArrays: Array[Tensor[_]]): AbstractModule[Activity, Activity, T]

  override def updateOutput(input: Table): Activity = {
    var rebuildModule = false
    if (module == null) {
      rebuildModule = true
    } else {
      indexes.map(i => input[Tensor[_]](i)).zip(initTensors).foreach(tensors => {
        if (tensors._1 != tensors._2) {
          rebuildModule = true
        }
      })
    }

    if (rebuildModule) {
      val l = input.length()
      indexes = configIndexes.map(getPositiveIndex(_, l))
      val tensors = indexes.map(i => input[Tensor[_]](i))
      initTensors = tensors.map(_.clone())
      module = build(tensors)
      dataIndexes = getDataIndexes(indexes, l)
      zeroGrads = tensors.map(t => t.emptyInstance().resizeAs(t))
    }

    realInput = if (dataIndexes.length == 1) {
      input[Tensor[_]](dataIndexes(0))
    } else {
      val t = T()
      dataIndexes.map(i => t.insert(input[Tensor[_]](i)))
      t
    }

    output = module.forward(realInput)
    output
  }

  private def getPositiveIndex(index: Int, length: Int): Int = {
    if (index > 0) index else length + index + 1
  }

  private def getDataIndexes(indexs: Array[Int], length: Int): Array[Int] = {
    (1 to length).filterNot(indexs.contains(_)).toArray
  }

  override def updateGradInput(input: Table, gradOutput: Activity): Table = {
    val realGradInput = module.updateGradInput(realInput, gradOutput)
    gradInput = T()
    var i = 0
    while(i < indexes.length) {
      gradInput(indexes(i)) = zeroGrads(i)
      i += 1
    }
    if (dataIndexes.length == 1) {
      gradInput(dataIndexes.head) = realGradInput
    } else {
      i = 0
      while (i < dataIndexes.length) {
        gradInput(dataIndexes(i)) = realGradInput.toTable.apply[Activity](i + 1)
        i += 1
      }
    }
    gradInput
  }

  override def accGradParameters(input: Table, gradOutput: Activity): Unit = {
    module.accGradParameters(realInput, gradOutput)
  }
}
