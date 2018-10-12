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

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.ops.Operation
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.{NumericWildcard, TensorNumeric}
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleSerializable, SerializeContext}
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.tf.TFTensorNumeric._

import org.tensorflow.example.{Example, Feature}

import scala.collection.JavaConverters._

import scala.reflect.ClassTag
import scala.reflect.runtime.universe

private[bigdl] class ParseExample[T: ClassTag](val nDense: Int,
  val tDense: Seq[TensorDataType],
  val denseShape: Seq[Array[Int]])
  (implicit ev: TensorNumeric[T])
  extends Operation[Table, Table, T] {

  type StringType = ByteString

  override def updateOutput(input: Table): Table = {
    require(input[Tensor[StringType]](1).size(1) == 1, "only support one example at a time")
    val serialized = input[Tensor[StringType]](1).valueAt(1)
    val denseKeys = Range(3, 3 + nDense).map(index => input(index).asInstanceOf[Tensor[StringType]])
      .map(_.value().toStringUtf8)
    val denseDefault = Range(3 + nDense, 3 + 2 * nDense)
      .map(index => input(index).asInstanceOf[Tensor[_]])


    val example = Example.parseFrom(serialized)

    val featureMap = example.getFeatures.getFeatureMap

    val outputs = denseDefault
      .zip(denseKeys)
      .zip(tDense).zip(denseShape).map { case (((default, key), tensorType), shape) =>
      if (featureMap.containsKey(key)) {
        val feature = featureMap.get(key)
        getTensorFromFeature(feature, tensorType, shape)
      } else {
        default
      }
    }

    for (elem <- outputs) {
      elem.asInstanceOf[Tensor[NumericWildcard]].addSingletonDimension()
      output.insert(elem)
    }
    output
  }

  private def getTensorFromFeature(feature: Feature,
    tensorType: TensorDataType,
    tensorShape: Array[Int]): Tensor[_] = {
    tensorType match {
      case LongType =>
        val values = feature.getInt64List.getValueList.asScala.map(_.longValue()).toArray
        Tensor(values, tensorShape)
      case FloatType =>
        val values = feature.getFloatList.getValueList.asScala.map(_.floatValue()).toArray
        Tensor(values, tensorShape)
      case StringType =>
        val values = feature.getBytesList.getValueList.asScala.toArray
        Tensor(values, tensorShape)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
  }
}

private[bigdl] class ParseSingleExample[T: ClassTag](val tDense: Seq[TensorDataType],
                                                     val denseKeys: Seq[ByteString],
                                               val denseShape: Seq[Array[Int]])
                                              (implicit ev: TensorNumeric[T])
  extends Operation[Table, Table, T] {

  type StringType = ByteString

  override def updateOutput(input: Table): Table = {
    val serialized = input[Tensor[StringType]](1).value()

    val example = Example.parseFrom(serialized)

    val featureMap = example.getFeatures.getFeatureMap

    val outputs = denseKeys
      .zip(tDense).zip(denseShape).map { case ((byteSKey, tensorType), shape) =>
      val key = byteSKey.toStringUtf8
      if (featureMap.containsKey(key)) {
        val feature = featureMap.get(key)
        getTensorFromFeature(feature, tensorType, shape)
      } else {
        None
      }
    }

    for (elem <- outputs) {
      output.insert(elem)
    }
    output
  }

  private def getTensorFromFeature(feature: Feature,
                                   tensorType: TensorDataType,
                                   tensorShape: Array[Int]): Tensor[_] = {
    tensorType match {
      case LongType =>
        val values = feature.getInt64List.getValueList.asScala.map(_.longValue()).toArray
        Tensor(values, tensorShape)
      case FloatType =>
        val values = feature.getFloatList.getValueList.asScala.map(_.floatValue()).toArray
        Tensor(values, tensorShape)
      case StringType =>
        val values = feature.getBytesList.getValueList.asScala.toArray
        Tensor(values, tensorShape)
      case t => throw new NotImplementedError(s"$t is not supported")
    }
  }
}

private[bigdl] object ParseExample extends ModuleSerializable {
  def apply[T: ClassTag](nDense: Int,
    tDense: Seq[TensorDataType],
    denseShape: Seq[Array[Int]])
    (implicit ev: TensorNumeric[T]): ParseExample[T] =
    new ParseExample[T](nDense, tDense, denseShape)

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap

    val nDense = DataConverter.getAttributeValue(context, attrMap.get("nDense")).
      asInstanceOf[Int]

    val tDense = DataConverter.getAttributeValue(context, attrMap.get("tDense")).
      asInstanceOf[Array[String]].map(toTensorType(_))

    val shapeSize = DataConverter.getAttributeValue(context, attrMap.get("shapeSize")).
      asInstanceOf[Int]

    val denseShape = new Array[Array[Int]](shapeSize)
    for (i <- 1 to shapeSize) {
      denseShape(i - 1) = DataConverter.getAttributeValue(context,
        attrMap.get(s"shapeSize_${i - 1}")).
        asInstanceOf[Array[Int]]
    }
    ParseExample[T](nDense, tDense.toSeq, denseShape.toSeq)
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    bigDLModelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {

    val parseExample = context.moduleData.module.asInstanceOf[ParseExample[T]]

    val nDenseBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, nDenseBuilder, parseExample.nDense,
      universe.typeOf[Int])
    bigDLModelBuilder.putAttr("nDense", nDenseBuilder.build)

    val tensorTypeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, tensorTypeBuilder,
      parseExample.tDense.toArray.map(fromTensorType(_)),
      universe.typeOf[Array[String]])
    bigDLModelBuilder.putAttr("tDense", tensorTypeBuilder.build)

    val shapeSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, shapeSizeBuilder,
      parseExample.denseShape.size,
      universe.typeOf[Int])
    bigDLModelBuilder.putAttr("shapeSize", shapeSizeBuilder.build)

    parseExample.denseShape.zipWithIndex.foreach(shape => {
      val shapeBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, shapeBuilder,
        parseExample.denseShape(shape._2),
        universe.typeOf[Array[Int]])
      bigDLModelBuilder.putAttr(s"shapeSize_${shape._2}", shapeBuilder.build)
    })

  }

  private def fromTensorType(ttype : TensorDataType): String = {
    ttype match {
      case LongType => "Long"
      case FloatType => "Float"
      case StringType => "String"
      case t => throw new NotImplementedError(s"$t is not supported")
    }
  }

  private def toTensorType(ttype : String): TensorDataType = {
    ttype match {
      case "Long" => LongType
      case "Float" => FloatType
      case "String" => StringType
    }
  }
}

private[bigdl] object ParseSingleExample extends ModuleSerializable {
  def apply[T: ClassTag](tDense: Seq[TensorDataType],
                         denseKeys: Seq[ByteString],
                         denseShape: Seq[Array[Int]])
                        (implicit ev: TensorNumeric[T]): ParseSingleExample[T] =
    new ParseSingleExample[T](tDense, denseKeys, denseShape)

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val attrMap = context.bigdlModule.getAttrMap

    val tDense = DataConverter.getAttributeValue(context, attrMap.get("tDense")).
      asInstanceOf[Array[String]].map(toTensorType(_))

    val denseKeysString = DataConverter.getAttributeValue(context,
      attrMap.get("denseKeys")).
      asInstanceOf[Array[String]]

    val denseKeys = new Array[ByteString](denseKeysString.length)

    (0 until denseKeysString.length).foreach(index => {
      val denseKeyBytes = denseKeysString(index).getBytes("utf-8")
      denseKeys(index) = ByteString.copyFrom(denseKeyBytes)
    })

    val shapeSize = DataConverter.getAttributeValue(context, attrMap.get("shapeSize")).
      asInstanceOf[Int]

    val denseShape = new Array[Array[Int]](shapeSize)
    for (i <- 1 to shapeSize) {
      denseShape(i - 1) = DataConverter.getAttributeValue(context,
        attrMap.get(s"shapeSize_${i - 1}")).
        asInstanceOf[Array[Int]]
    }
    ParseSingleExample[T](tDense.toSeq, denseKeys.toSeq, denseShape.toSeq)
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
    bigDLModelBuilder: BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val parseSingleExample = context.moduleData.module.asInstanceOf[ParseSingleExample[T]]

    val tensorTypeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, tensorTypeBuilder,
      parseSingleExample.tDense.toArray.map(fromTensorType(_)),
      universe.typeOf[Array[String]])
    bigDLModelBuilder.putAttr("tDense", tensorTypeBuilder.build)

    val denseKeys = parseSingleExample.denseKeys.toArray

    val denseKeysString = new Array[String](denseKeys.length)

    (0 until denseKeys.length).foreach(index => {
      denseKeysString(index) = new String(denseKeys(index).toByteArray, "utf-8")
    })

    val denseKeysBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, denseKeysBuilder,
      denseKeysString,
      universe.typeOf[Array[String]])
    bigDLModelBuilder.putAttr(s"denseKeys", denseKeysBuilder.build)
    val shapeSizeBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, shapeSizeBuilder,
      parseSingleExample.denseShape.size,
      universe.typeOf[Int])
    bigDLModelBuilder.putAttr("shapeSize", shapeSizeBuilder.build)

    parseSingleExample.denseShape.zipWithIndex.foreach(shape => {
      val shapeBuilder = AttrValue.newBuilder
      DataConverter.setAttributeValue(context, shapeBuilder,
        parseSingleExample.denseShape(shape._2),
        universe.typeOf[Array[Int]])
      bigDLModelBuilder.putAttr(s"shapeSize_${shape._2}", shapeBuilder.build)
    })

  }

  private def fromTensorType(ttype : TensorDataType): String = {
    ttype match {
      case LongType => "Long"
      case FloatType => "Float"
      case StringType => "String"
      case t => throw new NotImplementedError(s"$t is not supported")
    }
  }

  private def toTensorType(ttype : String): TensorDataType = {
    ttype match {
      case "Long" => LongType
      case "Float" => FloatType
      case "String" => StringType
    }
  }
}
