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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer.{ClassTagMapper, TensorNumericMapper}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import scala.util.Try

/**
 *
 */
object ModuleInitializer {

  def init[T: ClassTag](state: Table)(implicit ev: TensorNumeric[T]): Module[T] = {
    val classPath = state.get[String]("name") match {
      case Some(name) if name.startsWith("com.intel.analytics.bigdl.nn") =>
        name
      case Some(name) =>
        "com.intel.analytics.bigdl.nn." + name
      case None =>
        throw new IllegalArgumentException("No param `name` found in state(Table)!")
    }

    val cls = Try(Class.forName(classPath))
      .getOrElse(
        throw new IllegalArgumentException(s"Can't parse Class Path: $classPath")
      )

    val clsSymbol = mirror.classSymbol(cls)
    val pcOrNot = clsSymbol
      .typeSignature.declaration(nme.CONSTRUCTOR)
      .asTerm.alternatives.collectFirst {
      case cc: MethodSymbol if cc.isPrimaryConstructor => cc
    }
    val pc = pcOrNot.getOrElse(
      throw new IllegalArgumentException("Can't find Primary Constructor!"))

    val userDefined = state.getState()
    val args = pc.asMethod.paramss.head.zipWithIndex.map {
      case (symbol, index) =>
        val term = symbol.asTerm
        val fieldName = term.name.toString

        if (userDefined.contains(fieldName)) {
          userDefined(fieldName)
        } else if (userDefined.contains(index + 1)) {
          userDefined(index + 1)
        } else if (term.isParamWithDefault) {
          cls.getMethod("$lessinit$greater$default$" + (index + 1).toString).invoke(cls)
        } else {
          throw new IllegalArgumentException(
            s"Param $fieldName is not defined neither in Table nor by Default!")
        }
    }

    var addIndex = args.length
    val typeArgs: List[Any] = pc.asMethod.paramss.tail.flatMap {
      _.map { symbol =>
        symbol.typeSignature match {
          case tpe if tpe <:< typeOf[ClassTag[_]] =>
            val typeInfo = tpe.toString
            val tagName = typeInfo.slice(typeInfo.indexOf("[") + 1, typeInfo.indexOf("]"))
            addIndex += 1
            if (tagName == "T") {
              ClassTagMapper(TensorNumericMapper(ev))
            } else if (userDefined.contains(tagName)) {
              ClassTagMapper(userDefined(tagName).toString)
            } else if (userDefined.contains(addIndex)) {
              ClassTagMapper(userDefined(addIndex).toString)
            } else {
              throw new IllegalArgumentException(
                s"Type Param $tagName other than T is not defined in Table!")
            }

          case tpe if tpe <:< typeOf[TensorNumeric[_]] =>
            val evName = symbol.asTerm.name.toString
            addIndex += 1
            if (evName == "ev") {
              ev
            } else if (userDefined.contains(evName)) {
              TensorNumericMapper(userDefined(evName).toString)
            } else if (userDefined.contains(addIndex)) {
              TensorNumericMapper(userDefined(addIndex).toString)
            } else {
              throw new IllegalArgumentException(
                s"ev Param $evName other than TensorNumeric[T] is not defined in Table!")
            }
        }
      }
    }

    val classMirror = mirror.reflectClass(clsSymbol)
    val instance = classMirror.reflectConstructor(pc)(args ::: typeArgs: _*)
    instance.asInstanceOf[Module[T]]
  }

  private lazy val mirror = runtimeMirror(getClass.getClassLoader)

}
