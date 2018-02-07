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
 *  ======== ModuleInitializer ========
 *  ModuleInitializer, a tool for initializing(reflecting) Modules with a Table.
 */
object ModuleInitializer {

  private val prefix = "com.intel.analytics.bigdl.nn."

  def init[T: ClassTag](state: Table)(implicit ev: TensorNumeric[T]): Module[T] = {
    val classPath = if (state.contains("name")) {
      prefix + state.get[String]("name").get
    } else {
      prefix + Try(state.get[String](1).get).getOrElse(
        throw new IllegalArgumentException("Can't parse Module Name")
      )
    }

    val cls = Try(Class.forName(classPath)).getOrElse(
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

    // get parameters of primary constructor(userDefined override default values)
    val userDefined = state.getState()
    val args = pc.asMethod.paramss.head.zipWithIndex.map {
      case (symbol, index) =>
        val term = symbol.asTerm
        val fieldName = term.name.toString

        if (userDefined.contains(fieldName)) {
          userDefined(fieldName)
        } else if (userDefined.contains(index + 2)) {
          userDefined(index + 2)
        } else if (term.isParamWithDefault) {
          cls.getMethod("$lessinit$greater$default$" + (index + 1).toString).invoke(cls)
        } else {
          throw new IllegalArgumentException(
            s"Param $fieldName is not defined neither in Table nor by Default!")
        }
    }

    var addIndex = args.length + 1

    // Make a Map[alias of Tag, type of Tag], in order to config TensorNumeric automatically.
    val tagMap = pc.asMethod.paramss.tail.flatMap {
      _.filter(_.typeSignature <:< typeOf[ClassTag[_]])
        .map { symbol =>
          val typeInfo = symbol.typeSignature.toString
          val tagName = typeInfo.slice(typeInfo.indexOf("[") + 1, typeInfo.indexOf("]"))
          addIndex += 1

          val tagValue =
            if (tagName == "T") {
              TensorNumericMapper(ev)
            } else if (userDefined.contains(tagName)) {
              userDefined(tagName).toString
            } else if (userDefined.contains(addIndex)) {
              userDefined(addIndex).toString
            } else {
              throw new IllegalArgumentException(
                s"Type Param $tagName other than T is not defined in Table!")
            }

          tagName -> tagValue
        }
    }.toMap

    // get Type Params <:< ClassTag
    val typeArgs: List[Any] = tagMap.map { case(_, v) => ClassTagMapper(v) }.toList

    // get Type Params <:< TensorNumeric
    val numericArgs = pc.asMethod.paramss.tail.flatMap {
      _.filter(_.typeSignature <:< typeOf[TensorNumeric[_]])
        .map { symbol =>
          val typeInfo = symbol.typeSignature.toString
          val tagName = typeInfo.slice(typeInfo.indexOf("[") + 1, typeInfo.indexOf("]"))
          addIndex += 1

          TensorNumericMapper(tagMap(tagName))
        }
    }

    val allArgs = args ::: typeArgs ::: numericArgs
    val classMirror = mirror.reflectClass(clsSymbol)
    val instance = classMirror.reflectConstructor(pc)(allArgs: _*)
    instance.asInstanceOf[Module[T]]
  }

  private lazy val mirror = runtimeMirror(getClass.getClassLoader)

}
