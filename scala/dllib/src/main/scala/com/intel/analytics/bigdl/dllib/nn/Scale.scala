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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag

/**
 * Scale is the combination of cmul and cadd
 * Computes the elementwise product of input and weight, with the shape of the weight "expand" to
 * match the shape of the input.
 * Similarly, perform a expand cdd bias and perform an elementwise add
 * @param size size of weight and bias
 * @tparam T Numeric type. Only support float/double now
 */
class Scale[T: ClassTag](val size: Array[Int])
  (implicit ev: TensorNumeric[T]) extends AbstractModule[Tensor[T], Tensor[T], T] {

  private[bigdl] var cmul = new CMul[T](size)
  private[bigdl] var cadd = new CAdd[T](size)

  /**
   * Computes the output using the current parameter set of the class and input. This function
   * returns the result which is stored in the output field.
   * @param input
   * @return
   */
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output = cadd.forward(cmul.forward(input))
    output
  }

  /**
   * Computing the gradient of the module with respect to its own input. This is returned in
   * gradInput. Also, the gradInput state variable is updated accordingly.
   * @param input
   * @param gradOutput
   * @return
   */
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = cmul.backward(cadd.output, cadd.backward(input, gradOutput))
    gradInput
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   * @return (Array of weights, Array of grad)
   */
  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(cmul.parameters()._1(0), cadd.parameters()._1(0)),
      Array(cmul.parameters()._2(0), cadd.parameters()._2(0)))
  }

  override def toString: String = "nn.Scale"

  override def computeOutputShape(inputShape: Shape): Shape = {
    val outputShape = cmul.computeOutputShape(inputShape)
    cadd.computeOutputShape(outputShape)
  }
}

object Scale extends ModuleSerializable {
  def apply[@specialized(Float, Double) T: ClassTag](size: Array[Int])
    (implicit ev: TensorNumeric[T]): Scale[T] = new Scale[T](size)

  override def doLoadModule[T: ClassTag](context : DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val scale = super.doLoadModule(context).asInstanceOf[Scale[T]]
    val attrMap = context.bigdlModule.getAttrMap
    val cmul = attrMap.get("cmul")
    scale.cmul = DataConverter.getAttributeValue(context, cmul).asInstanceOf[CMul[T]]
    val cadd = attrMap.get("cadd")
    scale.cadd = DataConverter.getAttributeValue(context, cadd).asInstanceOf[CAdd[T]]
    scale
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                            scaleBuilder : BigDLModule.Builder)
                                           (implicit ev: TensorNumeric[T]) : Unit = {
    val scale = context.moduleData.module.asInstanceOf[Scale[T]]
    super.doSerializeModule(context, scaleBuilder)

    val cmulBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, cmulBuilder,
      scale.cmul, ModuleSerializer.abstractModuleType)
    scaleBuilder.putAttr("cmul", cmulBuilder.build)

    val caddBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, caddBuilder,
      scale.cadd, ModuleSerializer.abstractModuleType)
    scaleBuilder.putAttr("cadd", caddBuilder.build)

  }
}
