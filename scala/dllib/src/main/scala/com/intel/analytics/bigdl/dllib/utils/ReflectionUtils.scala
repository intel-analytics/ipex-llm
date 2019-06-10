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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.MklInt8Convertible
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.intermediate.{IRElement, IROperator}

import scala.reflect.ClassTag
import scala.reflect.runtime.{currentMirror, universe}



private[bigdl] object ReflectionUtils {

  private val runtimeMirror = universe.runtimeMirror(getClass.getClassLoader)

  /* -------------------------- External API -------------------------- */
  /**
   * Instantiate an object from the target Class with given parameter values,
   * assign default values if parameters' value is missing.
   * @param target target class
   * @param valueMap param's name as key, param's value as value
   * @param tags class tag
   * @param numerics type of tensor numeric the class is using
   * @return
   */
  def reflection(target: Class[_], valueMap: Map[String, AnyRef],
      tags: Array[ClassTag[_]], numerics: Array[TensorNumeric[_]]): Object = {

    val ctor = getPrimCtorMirror(target)
    val paramList = getPrimCtorParamList(target)
    val instanceMirror = getInstanceMirror(target)
    val (tagsIter, numericsIter) = (tags.toIterator, numerics.toIterator)

    val args = paramList.map {
      param =>
        val typeSig = param.symbol.typeSignature
        if (isClassTag(typeSig)) {
          tagsIter.next()
        } else if (isTensorNumeric(typeSig)) {
          numericsIter.next()
        } else {
          val pname = param.symbol.name.decodedName.toString
          //
          valueMap.getOrElse(pname, getPrimCtorDefaultParamValue(
            instanceMirror,
            param.symbol,
            param.index
          ))
        }
    }

    ctor.apply(args : _*).asInstanceOf[Object]
  }


  // TODO: to be refined, naming is confusing
  /**
   * Create a Module object of target Class by mirroring the given IR element
   * @param source the source IR element
   * @param target the target class we want to make an object from
   * @tparam T
   * @return the target instance of type Module
   */
  def reflectFromIR[T : ClassTag](source: IRElement[T], target: Class[_]): Module[T] = {
    val nameAndValues = getFieldNameAndValues(source.getOp())
    val (tags, numerics) = source.getOp().getClassTagNumerics()

    val blasLayer = reflection(target, nameAndValues, tags, numerics)
      .asInstanceOf[Module[T]]

    if (blasLayer.parameters() != null) {
      val params = blasLayer.getParameters()
      val params2 = source.getParameters()
      if (params2._1 != null) {
        params._1.copy(params2._1)
        source.setWeights(params._1)
      }
      if (params2._2 != null) {
        params._2.copy(params2._2)
        source.setGradWeights(params._2)
      }
    }

    if (source.getName() != "") blasLayer.setName(source.getName())
    if (blasLayer.isInstanceOf[MklInt8Convertible]) {
      setScales(source, blasLayer.asInstanceOf[MklInt8Convertible])
    }

    blasLayer
  }


  /**
   * Create an IR element object of target Class by mirroring given source Module
   * @param source the source Module we want to mirror
   * @param target the class of target IR element we want to create
   * @tparam T
   * @return
   */
  def reflectToIR[T: ClassTag](source: Module[T], target: Class[_]): IRElement[T] = {
    val nameAndValues = getFieldNameAndValues(source)
    val (tags, numerics) = source.getClassTagNumerics()
    val op = ReflectionUtils.reflection(target, nameAndValues,
      tags, numerics).asInstanceOf[IROperator[T]]
    val weightsAndBias =
      if (source.parameters() != null) source.getParameters() else (null, null)
    val element = IRElement[T](
      source.getName(), op, weights = weightsAndBias._1, gradWeights = weightsAndBias._2)
    if (source.isInstanceOf[MklInt8Convertible]) {
      setScales(source.asInstanceOf[MklInt8Convertible], element)
    }
    element
  }


  /**
   * Get the primary class constructor of input class
   * @param cls
   * @tparam T
   * @return
   */
  def getPrimCtorMirror[T : ClassTag](cls : Class[_]): universe.MethodMirror = {

    val clsSymbol = runtimeMirror.classSymbol(cls)
    val cm = runtimeMirror.reflectClass(clsSymbol)
    // to make it compatible with both 2.11 and 2.10
    val ctorCs = clsSymbol.toType.declaration(universe.nme.CONSTRUCTOR)
    val primary: Option[universe.MethodSymbol] = ctorCs.asTerm.alternatives.collectFirst {
      case cstor if cstor.asInstanceOf[universe.MethodSymbol].isPrimaryConstructor =>
        cstor.asInstanceOf[universe.MethodSymbol]
    }
    cm.reflectConstructor(primary.get)

  }


  def findClass(name: String): Class[_] = {
    try {
      Class.forName(name)
    } catch {
      case ex: ClassNotFoundException => null
      case e: Throwable => throw e
    }
  }


  // TODO: this method should be moved to a more appropriate place
  // put scales in fromEle to toELe
  def setScales[T: ClassTag](fromEle: MklInt8Convertible,
                                     toELe: MklInt8Convertible): Unit = {
    toELe.setInputScales(fromEle.getInputScales())
    toELe.setOutputScales(fromEle.getOutputScales())
    toELe.setWeightScales(fromEle.getWeightScales())

    toELe.setInputDimMask(fromEle.getInputDimMask(), true)
    toELe.setOutputDimMask(fromEle.getOutputDimMask(), true)
    toELe.setWeightDimMask(fromEle.getWeightDimMask(), true)
  }


  /* -------------------------- Internal API -------------------------- */
  /**
   * Get key value map from input object,
   * field name of the object as key, its reference as value
   * @param o input object
   * @return A map which field name as key and field refernece as value
   */
  private def getFieldNameAndValues(o: Object): Map[String, AnyRef] = {
    val c = o.getClass
    var fields = c.getDeclaredFields
    val superFields = c.getSuperclass.getDeclaredFields

    fields = fields ++ superFields

    val values = fields.map {
      field =>
        field.setAccessible(true)
        (field.getName, field.get(o))
    }.toMap

    values
  }


  /**
   * Get instance mirror of the input target Class if it has been defined as a case class or
   * has a companion object, otherwise it returns universe.NoSymbol
   * @param target
   * @return InstanceMirror, if the class has no companion then universe.NoSymbol
   */
  private def getInstanceMirror(target: Class[_]): universe.InstanceMirror = {
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

    instanceMirror
  }

  /**
   * Get primary constructor parameter list of the target Class
   * @param target
   * @return
   */
  private def getPrimCtorParamList(target: Class[_]): List[CtorParam] = {

    val ctorMirror = getPrimCtorMirror(target)
    val ctorParamSymbols = ctorMirror.symbol.paramss

    val ctorParamList = ctorParamSymbols.flatten.zipWithIndex.map {
      case (param, index) =>
        CtorParam(index, param)
    }

    ctorParamList
  }

  /**
   * Check given type signature is of ClassTag or not
   * @param typeSig
   * @return
   */
  private def isClassTag(typeSig: universe.Type): Boolean = {
    typeSig <:< universe.typeOf[ClassTag[_]] ||
      typeSig.typeSymbol == universe.typeOf[ClassTag[_]].typeSymbol
  }

  /**
   * Check given type signature is of TensorNumeric or not
   * @param typeSig
   * @return
   */
  private def isTensorNumeric(typeSig: universe.Type): Boolean = {
    typeSig <:< universe.typeOf[TensorNumeric[_]] ||
      typeSig.typeSymbol == universe.typeOf[TensorNumeric[_]].typeSymbol
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

private case class CtorParam(index: Int, symbol: universe.Symbol)

