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

package com.intel.analytics.bigdl.utils.intermediate

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.MklInt8Convertible
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializer._

import scala.collection.mutable
import scala.reflect.{ClassTag, ManifestFactory}
import scala.reflect.runtime._

private[bigdl] object ReflectionUtils {

  private def getFieldNameAndValues(o: Object): mutable.HashMap[String, AnyRef] = {
    val c = o.getClass
    var fields = c.getDeclaredFields
    val superFields = c.getSuperclass.getDeclaredFields
    fields = fields ++ superFields

    val values = new mutable.HashMap[String, AnyRef]()
    fields.foreach(field => {
      field.setAccessible(true)
      values(field.getName) = field.get(o)
    })
    values
  }

  // create layer2 object form layer1
  private def reflection(layer1: Object, layer2: Class[_],
     tags: Array[ClassTag[_]], numerics: Array[TensorNumeric[_]]) : Object = {
    val nameAndValues = getFieldNameAndValues(layer1)
    val constructorMirror = getCostructorMirror(layer2)
    val constructorFullParams = constructorMirror.symbol.paramss
    val args = new Array[Object](constructorFullParams.map(_.size).sum)

    val tagIter = tags.iterator
    val numericIter = numerics.iterator
    var i = 0
    constructorFullParams.foreach(map => {
      map.foreach(param => {
        val name = param.name.decodedName.toString
        val ptype = param.typeSignature
        if (ptype <:< universe.typeOf[ClassTag[_]]||
          ptype.typeSymbol == universe.typeOf[ClassTag[_]].typeSymbol) {
          require(tagIter.hasNext, "If your module contains multiple class tags, " +
            s"do you forget to override getClassTagNumerics method ${layer1}")
          args(i) = tagIter.next()
        } else if (ptype <:< universe.typeOf[TensorNumeric[_]]
          || ptype.typeSymbol == universe.typeOf[TensorNumeric[_]].typeSymbol) {
          args(i) = numericIter.next()
        } else {
          val value = nameAndValues.get(name).getOrElse(null)
          args(i) = value
        }
        i += 1
      })
    })
    constructorMirror.apply(args : _*).asInstanceOf[Object]
  }

  // create Module form IRElement
  def reflectFromIR[T : ClassTag](layer: IRElement[T], cls: Class[_]) : Module[T] = {
    val (tags, numerics) = layer.getOp().getClassTagNumerics()
    val blasLayer = ReflectionUtils.reflection(layer.getOp(), cls, tags, numerics)
      .asInstanceOf[Module[T]]

    if (blasLayer.parameters() != null) {
      val params = blasLayer.getParameters()
      val params2 = layer.getParameters()
      if (params2._1 != null) {
        params._1.copy(params2._1)
        layer.setWeights(params._1)
      }
      if (params2._2 != null) {
        params._2.copy(params2._2)
        layer.setGradWeights(params._2)
      }
    }

    if (layer.getName() != "") blasLayer.setName(layer.getName())
    if (blasLayer.isInstanceOf[MklInt8Convertible]) {
      setScales(layer, blasLayer.asInstanceOf[MklInt8Convertible])
    }

    blasLayer
  }

  // create IRElement form Module
  def reflectToIR[T: ClassTag](layer: Module[T], cls: Class[_]) : IRElement[T] = {
    val (tags, numerics) = layer.getClassTagNumerics()
    val op = ReflectionUtils.reflection(layer, cls, tags, numerics).asInstanceOf[IROperator[T]]
    val weightsAndBias =
      if (layer.parameters() != null) layer.getParameters() else (null, null)
    val element = IRElement[T](
      layer.getName(), op, weights = weightsAndBias._1, gradWeights = weightsAndBias._2)
    if (layer.isInstanceOf[MklInt8Convertible]) {
      setScales(layer.asInstanceOf[MklInt8Convertible], element)
    }
    element
  }

  // put scales in fromEle to toELe
  private[intermediate] def setScales[T: ClassTag](fromEle: MklInt8Convertible,
                                     toELe: MklInt8Convertible): Unit = {
    toELe.setInputScales(fromEle.getInputScales())
    toELe.setOutputScales(fromEle.getOutputScales())
    toELe.setWeightScales(fromEle.getWeightScales())

    toELe.setInputDimMask(fromEle.getInputDimMask())
    toELe.setOutputDimMask(fromEle.getOutputDimMask())
    toELe.setWeightDimMask(fromEle.getWeightDimMask())
  }

  def findClass(name: String): Class[_] = {
    try {
      Class.forName(name)
    } catch {
      case ex: ClassNotFoundException => null
      case e: Throwable => throw e
    }
  }
}
