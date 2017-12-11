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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{DataConverter, DeserializeContext, ModuleSerializable, SerializeContext}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe

/**
 * Extracts a strided slice from a tensor.
 * @param sliceSpecs Array(dim, begin_index, end_index, stride)
 */
@SerialVersionUID(4436600172725317184L)
private[bigdl] class StrideSlice[T: ClassTag](val sliceSpecs: Array[(Int, Int, Int, Int)])
                (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  require(sliceSpecs.map(_._4 == 1).reduce(_ && _), "only support stride 1 for now")

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var tmp = input
    var i = 0
    while(i < sliceSpecs.length) {
      tmp = tmp.narrow(sliceSpecs(i)._1, sliceSpecs(i)._2, sliceSpecs(i)._3 - sliceSpecs(i)._2)
      i += 1
    }
    output.resizeAs(tmp)
    output.copy(tmp)
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    gradInput.zero()
    var tmp = gradInput
    var i = 0
    while(i < sliceSpecs.length) {
      tmp = tmp.narrow(sliceSpecs(i)._1, sliceSpecs(i)._2, sliceSpecs(i)._3 - sliceSpecs(i)._2)
      i += 1
    }
    tmp.copy(gradOutput)
    gradInput
  }

}

private[bigdl] object StrideSlice extends ModuleSerializable {
  def apply[T: ClassTag](sliceSpecs: Array[(Int, Int, Int, Int)])
                        (implicit ev: TensorNumeric[T]) : StrideSlice[T] = {
    new StrideSlice[T](sliceSpecs)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap
    // val module = super.doLoadModule(context)

    val sliceLen = attrMap.get("sliceLen").getInt32Value

    val specs = new Array[(Int, Int, Int, Int)](sliceLen)
    for (i <- 0 until sliceLen) {
      val spec = attrMap.get(s"spec_$i")
      val lst = DataConverter.
        getAttributeValue(context, spec).asInstanceOf[Array[Int]]
      specs(i) = (lst(0), lst(1), lst(2), lst(3))
    }
    StrideSlice[T](specs)
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              recurrentBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {

    val strideSlice = context.moduleData.module.asInstanceOf[StrideSlice[T]]

    val sliceSpecs = strideSlice.sliceSpecs

    val lengthBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context,
      lengthBuilder, sliceSpecs.length,
      universe.typeOf[Int])
    recurrentBuilder.putAttr("sliceLen", lengthBuilder.build)

    sliceSpecs.zipWithIndex.foreach(pair => {
      val specBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context,
        specBuilder, Array[Int](pair._1._1, pair._1._2, pair._1._3, pair._1._4),
        universe.typeOf[Array[Int]])
      recurrentBuilder.putAttr(s"spec_${pair._2}", specBuilder.build)
    })
  }
}

