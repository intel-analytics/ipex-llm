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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, SerializeContext}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe

private[bigdl] trait RandomNode

class RandomUniform[T: ClassTag, D: ClassTag](
  val minVal: Double, val maxVal: Double, val seed: Option[Int] = None
)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Tensor[Int], Tensor[D], T] with RandomNode {

  if (seed.isDefined) {
    RandomGenerator.RNG.setSeed(seed.get)
  }

  output = Activity.allocate[Tensor[D], D]()

  override def updateOutput(input: Tensor[Int]): Tensor[D] = {
    require(input.nDimension() == 1, "the shape should be a one-dimensional tensor.")

    val shape = input.storage().toArray
    output.resize(shape).rand(
      minVal,
      maxVal)

    output
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }
}

object RandomUniform extends ModuleSerializable {
  def apply[T: ClassTag, D: ClassTag](
                                       minVal: Double,
                                       maxVal: Double,
                                       seed: Option[Int] = None)
                                     (implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]):
  Operation[Activity, Activity, T]
  = ModuleToOperation[T](new RandomUniform[T, D](minVal, maxVal, seed))

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    bigDLModelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val randomUniform = context.moduleData.module.asInstanceOf[RandomUniform[T, _]]

    val minValBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, minValBuilder, randomUniform.minVal,
      universe.typeOf[Double])
    bigDLModelBuilder.putAttr("minVal", minValBuilder.build)

    val maxValBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, maxValBuilder, randomUniform.maxVal,
      universe.typeOf[Double])
    bigDLModelBuilder.putAttr("maxVal", maxValBuilder.build)

    if (randomUniform.seed.isDefined) {
      val seedBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, seedBuilder, randomUniform.seed.get,
        universe.typeOf[Int])
      bigDLModelBuilder.putAttr("seed", seedBuilder.build)
    }
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val attrMap = context.bigdlModule.getAttrMap
    val minVal = attrMap.get("minVal").getDoubleValue
    val maxVal = attrMap.get("maxVal").getDoubleValue
    var seed : Option[Int] = None
    if (attrMap.containsKey("seed")) {
      seed = Option[Int](attrMap.get("seed").getInt32Value)
    }
    RandomUniform(minVal, maxVal, seed)
  }
}

