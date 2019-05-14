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
  private def reflection(source: Object, target: Class[_],
     tags: Array[ClassTag[_]], numerics: Array[TensorNumeric[_]]) : Object = {
    val nameAndValues = getFieldNameAndValues(source)
    val constructorMirror = getCostructorMirror(target)
    val constructorFullParams = constructorMirror.symbol.paramss
    val args = new Array[Object](constructorFullParams.map(_.size).sum)

    val tagIter = tags.iterator
    val numericIter = numerics.iterator

    val clsMirror = universe.runtimeMirror(target.getClassLoader)
    val clsSymbol = clsMirror.classSymbol(target)

    /*
    https://www.scala-lang.org/api/2.10.7/#scala.reflect.api.Symbols$Symbol
    this line tries to get companion object of the class;
    through the companion, default values can be accessed by calling
    some static methods created by scala compiler, however it does not work when
    the class is not a case class or has not defined a companion, which in this case,
    calling companionSymbol returns universe.NoSymbol
    */
    val companionSymbol = clsSymbol.companionSymbol

    val instanceMirror = companionSymbol match {
      case universe.NoSymbol => null
      case _ =>
        val compnInst = currentMirror.reflectModule(clsSymbol.companionSymbol.asModule).instance
        clsMirror.reflect(compnInst)
    }

    constructorFullParams.flatten.zipWithIndex.map {
      case (param, idx) =>
        val pname = param.name.decodedName.toString
        val ptypesig = param.typeSignature
        if (ptypesig <:< universe.typeOf[ClassTag[_]]||
          ptypesig.typeSymbol == universe.typeOf[ClassTag[_]].typeSymbol) {
          require(tagIter.hasNext, "If your module contains multiple class tags, " +
            "do you forget to override getClassTagNumerics method")
          args(idx) = tagIter.next
        } else if (ptypesig <:< universe.typeOf[TensorNumeric[_]]
          || ptypesig.typeSymbol == universe.typeOf[TensorNumeric[_]].typeSymbol) {
          args(idx) = numericIter.next
        } else {
          val pvalue = if (nameAndValues.contains(pname)) { // for existing parameters
            nameAndValues.get(pname).getOrElse(null)
          } else { // parameter not found, get its default value
            getPrimCtorDefaultParamValue(instanceMirror, param, idx)
          }
          args(idx) = pvalue
        }
    }
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

    toELe.setInputDimMask(fromEle.getInputDimMask(), true)
    toELe.setOutputDimMask(fromEle.getOutputDimMask(), true)
    toELe.setWeightDimMask(fromEle.getWeightDimMask(), true)
  }

  def findClass(name: String): Class[_] = {
    try {
      Class.forName(name)
    } catch {
      case ex: ClassNotFoundException => null
      case e: Throwable => throw e
    }
  }

  /**
   * Get class primary consturctor's default parameter value by index
   * @param instMirror instance mirror object of the class companion object
   * @param paramSymbol symbol object of the target parameter with default value
   * @param index the index of parameter in the class primary constructor
   * @return AnyRef which is compatible with java Object
   */
  private def getPrimCtorDefaultParamValue(instMirror: universe.InstanceMirror,
                                           paramSymbol: universe.Symbol,
                                           index: Int): AnyRef = {
    if (paramSymbol == null || paramSymbol == universe.NoSymbol ||
      instMirror == null || index < 0) {
      return None
    }

    if (!paramSymbol.asTerm.isParamWithDefault) { // param has no default value
      None
    } else {
      val instTypeSig = instMirror.symbol.typeSignature
      val methodName = getCtorDefaultParamMethodByIndex(index)
      val methodSymbol = instTypeSig.member(universe.newTermName(methodName))
      if (methodSymbol == universe.NoSymbol) { // method not found
        None
      }
      else {
        // make the method call using reflection
        // need to cast it as AnyRef to be compatible with Java Object type
        instMirror.reflectMethod(methodSymbol.asMethod).apply().asInstanceOf[AnyRef]
      }
    }
  }

  /**
   * get string name of the method, which returns default value of the i-th parameter
   * Reference:
   * https://stackoverflow.com/questions/39657211/scala-class-constructors-default-argument-naming
   * @param i parameter index in primary constructor
   * @return method name in string, calling this method returns default value of i-th parameter
   */
  private def getCtorDefaultParamMethodByIndex(i: Int): String = {
    s"$$lessinit$$greater$$default$$${i + 1}"
  }
}

