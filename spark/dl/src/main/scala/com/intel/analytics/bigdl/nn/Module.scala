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

import java.nio.ByteOrder

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.bigdl.utils.serializer.{ClassTagMapper, ModuleLoader, TensorNumericMapper}
import com.intel.analytics.bigdl.utils.tf.{Session, TensorflowLoader}
import com.intel.analytics.bigdl.utils.{File, Table}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._
import scala.util.Try

object Module {

  private val moduleClassPathPrefix = "com.intel.analytics.bigdl.nn."

  private lazy val mirror = runtimeMirror(getClass.getClassLoader)

  /**
   * <h6>Initialize a module dynamically</h6>
   * Initialize a module dynamically by calling its primary constructor by runtime reflection.
   * <br><br>
   * A model is defined by its `ClassPath`, which according to state("name"). A optional `ClassPath`
   * prefix, `[com.intel.analytics.bigdl.nn.]`, is provided for simplifying model definitions.
   * <br><br>
   * Parameters of primary constructor can be set by state(state(`ParamKey`)=`ParamValue`).
   * `ParamKey` can be either the literal name of param or the index in param list(starting from 1).
   * It is unnecessary to set all params in state. For params with default value(in constructor),
   * their defaults will work when they are not set in state, otherwise their defaults will be
   * overridden by settings.
   * <br><br>
   * Type parameters can be set with their type tag(as `ParamKey`, such as T,D) and literal
   * names of types(as `ParamValue`, such as "Double","Int").
   * Only primitive types(Float/Double/Char/Boolean/String/Int/Long) are supported.
   * `TensorNumeric` of type parameters will be generated automatically if necessary.
   * The main type parameter `T`(of Module[T]) will be set automatically.
   * <br><br>
   * For now, nested states are not supported.
   *
   * @param state args of the model stored in key-value way
   * @param withClassPathPrefix whether adding prefix while parsing Classpath, default: `true`
   * @tparam T type parameter of Module, with TensorNumeric of this type
   */
  def apply[T: ClassTag](
      state: Table,
      withClassPathPrefix: Boolean = true)
    (implicit ev: TensorNumeric[T]): Module[T] = {

    // TODO: support parsing Regularizer/InitMethod and other usual structures
    // TODO: support nested state
    val prefix = if (withClassPathPrefix) moduleClassPathPrefix else ""
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

  /**
   * Load model from path.
   *
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @tparam T numeric type
   * @return model loaded from path
   */
  @deprecated("Java based serialization not recommended any more, please use loadModule instead")
  def load[T: ClassTag](path : String) : AbstractModule[Activity, Activity, T] = {
    File.load[AbstractModule[Activity, Activity, T]](path)
  }

  /**
   * Load model from path.
   *
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param weightPath : where weight is stored
   * @tparam T numeric type
   * @return model loaded from path
   */
  def loadModule[T: ClassTag](path : String,
      weightPath : String = null)(implicit ev: TensorNumeric[T])
  : AbstractModule[Activity, Activity, T] = {
    ModuleLoader.loadFromFile(path, weightPath)
  }

  def loadTorch[T: ClassTag](path : String) : AbstractModule[Activity, Activity, T] = {
    File.loadTorch[AbstractModule[Activity, Activity, T]](path)
  }

  @deprecated
  def loadCaffe[T: ClassTag](model: AbstractModule[Activity, Activity, T],
      defPath: String, modelPath: String, matchAll: Boolean = true)(
      implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    CaffeLoader.load[T](model, defPath, modelPath, matchAll)
  }

  /**
   * Loaf caffe trained model from prototxt and weight files
   * @param defPath  caffe model definition file path
   * @param modelPath caffe model binary file containing weight and bias
   */
  def loadCaffeModel[T: ClassTag](defPath: String, modelPath: String)(
      implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    CaffeLoader.loadCaffe[T](defPath, modelPath)._1
  }
  /**
   * Load tensorflow model from its saved protobuf file.
   * @param graphFile where is the protobuf model file
   * @param inputs input node names
   * @param outputs output node names, the output tensor order is same with the node order
   * @param byteOrder byte order in the tensorflow file. The default value is little endian
   * @param binFile where is the model variable file
   * @return BigDL model
   */
  def loadTF[T: ClassTag](graphFile: String, inputs: Seq[String], outputs: Seq[String],
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
      binFile: Option[String] = None)(
      implicit ev: TensorNumeric[T]): Module[T] = {

    TensorflowLoader.load(graphFile, inputs, outputs, byteOrder, binFile)
  }

  /**
   * Load tensorflow checkpoints
   * @param graphFile
   * @param binFile
   * @tparam T
   * @return
   */
  def tensorflowCheckpoints[T: ClassTag](graphFile: String, binFile: String,
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN)(implicit ev: TensorNumeric[T]): Session[T] = {
    TensorflowLoader.checkpoints(graphFile, binFile, byteOrder)
  }

  def flatten[@specialized(Float, Double) T: ClassTag](parameters: Array[Tensor[T]])(
      implicit ev: TensorNumeric[T]): Tensor[T] = {
    val compactedTensor = isCompact(parameters)
    if (compactedTensor != null) {
      return compactedTensor
    }
    var i = 0
    var length = 0
    while (i < parameters.length) {
      require(parameters(i).isContiguous(), "parameters should be contiguous")
      length += parameters(i).nElement()
      i += 1
    }

    val result = Tensor[T](length)
    val resultStorage = result.storage()

    i = 0
    var offset = 0
    while (i < parameters.length) {
      System.arraycopy(parameters(i).storage().array(), parameters(i).storageOffset() - 1,
        resultStorage.array(), offset, parameters(i).nElement())
      parameters(i).set(resultStorage, offset + 1, parameters(i).size(), parameters(i).stride())
      offset += parameters(i).nElement()
      i += 1
    }

    result
  }

  def isCompact[@specialized(Float, Double) T: ClassTag](parameters: Array[Tensor[T]])(
      implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(parameters.length > 0,
      "The length of paramters should >= 0" +
        "parameter length" +
        s" ${parameters.length}")
    var i = 1
    val storage = parameters(0).storage()
    var length = parameters(0).nElement()
    val offset = parameters(0).storageOffset()
    // make sure parameters is shared and contiguous
    while (i < parameters.length) {
      if (!storage.eq(parameters(i).storage())) {
        return null
      }
      if (offset + length != parameters(i).storageOffset()) {
        return null
      }
      length += parameters(i).nElement()
      i += 1
    }

    Tensor(storage, offset, Array(length))
  }
}
