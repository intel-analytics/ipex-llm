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
package com.intel.analytics.bigdl.utils.tf

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.Charset

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor, TensorDataType}
import org.tensorflow.framework.AttrValue.ListValue
import org.tensorflow.framework._
import org.tensorflow.framework.TensorShapeProto.Dim

import scala.reflect.{ClassTag, classTag}

/**
 * Tensorflow data format. It is mostly applied in processing image type data
 */
sealed trait TensorflowDataFormat {
  def value : AttrValue
}

object TensorflowDataFormat {
  /**
   * Store the image data in tensor as batch, height, width, channel
   */
  object NHWC extends TensorflowDataFormat {
    private val v = AttrValue.newBuilder().setS(ByteString
      .copyFrom("NHWC", Charset.defaultCharset())).build()

    override def value: AttrValue = v
  }

  /**
   * Store the image data in tensor as batch, channel, height, width
   */
  object NCHW extends TensorflowDataFormat {
    private val v = AttrValue.newBuilder().setS(ByteString
      .copyFrom("NCHW", Charset.defaultCharset())).build()

    override def value: AttrValue = v
  }
}

/**
 * Tensorflow padding type
 */
sealed trait PaddingType {
  def value : AttrValue
}

object PaddingType {

  object PADDING_SAME extends PaddingType {
    private val v = AttrValue.newBuilder().setS(ByteString
      .copyFrom("SAME", Charset.defaultCharset())).build()

    override def value: AttrValue = v
  }

  object PADDING_VALID extends PaddingType {
    private val v = AttrValue.newBuilder().setS(ByteString
      .copyFrom("VALID", Charset.defaultCharset())).build()

    override def value: AttrValue = v
  }
}

object Tensorflow {
  /**
   * Generate a placeholder tensorflow protobuf node
   * @param dtype numeric type
   * @param shape shape
   * @param name node name
   * @return
   */
  def placeholder(dtype: TensorDataType, shape: Seq[Int], name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Placeholder")
      .putAttr("dtype", typeAttr(dtype))
      .putAttr("shape", shapeAttr(shape))
      .build()
  }

  /**
   * Generate a const tensorflow protobuf node
   * @param value
   * @param name
   * @return
   */
  def const[T: ClassTag](value : Tensor[T], name : String, byteOrder: ByteOrder,
                         isScalar: Boolean = false, dataType: DataType = null): NodeDef = {
    val dtype = if (dataType == null) {
      if (value.getType() == DoubleType) {
        DataType.DT_DOUBLE
      } else {
        DataType.DT_FLOAT
      }
    } else {
      dataType
    }

    NodeDef.newBuilder()
      .setName(name)
      .setOp("Const")
      .putAttr("dtype", AttrValue.newBuilder().setType(dtype).build())
      .putAttr("value", tensorAttr(value, dtype, byteOrder, isScalar))
      .build()
  }

  /**
   * Generate an identity tensorflow protobuf node
   * @param input
   * @param name
   * @return
   */
  def identity(input : NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Identity")
      .addInput(input.getName)
      .putAttr("T", getDataType(input))
      .build()
  }

  /**
   * Generate a matmul tensorflow protobuf node
   * @param a
   * @param b
   * @param name
   * @param transposeA
   * @param transposeB
   * @return
   */
  def matmul(a: NodeDef, b: NodeDef, name: String,
             transposeA: Boolean = false, transposeB: Boolean = false): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("MatMul")
      .addInput(a.getName)
      .addInput(b.getName)
      .putAttr("T", getDataType(a))
      .putAttr("transpose_a", booleanAttr(transposeA))
      .putAttr("transpose_b", booleanAttr(transposeB))
      .build()
  }

  /**
   * Generate a biasAdd tensorflow protobuf node
   * @param value
   * @param bias
   * @param dataFormat
   * @param name
   * @return
   */
  def biasAdd(value: NodeDef, bias: NodeDef, dataFormat: TensorflowDataFormat,
              name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("BiasAdd")
      .addInput(value.getName)
      .addInput(bias.getName)
      .putAttr("T", getDataType(value))
      .putAttr("data_format", dataFormat.value)
      .build()
  }

  /**
   * Generate a relu tensorflow protobuf node
   * @param features
   * @param name
   * @return
   */
  def relu(features: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Relu")
      .addInput(features.getName)
      .putAttr("T", getDataType(features))
      .build()
  }

  def conv2D(input: NodeDef, filter: NodeDef, sW: Int, sH: Int, kW: Int, kH: Int, pW: Int, pH: Int,
             dataFormat: TensorflowDataFormat, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Conv2D")
      .addInput(input.getName)
      .addInput(filter.getName)
      .putAttr("T", getDataType(input))
      .putAttr("data_format", dataFormat.value)
      .putAttr("padding", getPaddingType(pW, pH, kW, kH, sW, sH).value)
      .putAttr("strides", strideAttr(sW, sH, dataFormat))
      .build()
  }

  def squeeze(input: NodeDef, axis: Seq[Int], name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Squeeze")
      .addInput(input.getName)
      .putAttr("T", getDataType(input))
      .putAttr("squeeze_dims", listIntAttr(axis))
      .build()
  }

  def tanh(input: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Tanh")
      .addInput(input.getName)
      .putAttr("T", getDataType(input))
      .build()
  }

  def reshape(tensor: NodeDef, shape: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Reshape")
      .addInput(tensor.getName)
      .addInput(shape.getName)
      .putAttr("T", getDataType(tensor))
      .putAttr("Tshape", getDataType(shape))
      .build()

  }

  def maxPool(value: NodeDef, kW: Int, kH: Int, pW: Int, pH: Int, sW: Int, sH: Int,
              dataFormat: TensorflowDataFormat, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("MaxPool")
      .addInput(value.getName)
      .putAttr("T", getDataType(value))
      .putAttr("data_format", dataFormat.value)
      .putAttr("ksize", kernelAttr(kW, kH, dataFormat))
      .putAttr("padding", getPaddingType(pW, pH, kW, kH, sW, sH).value)
      .putAttr("strides", strideAttr(sW, sH, dataFormat))
      .build()
  }

  def avgPool(value: NodeDef, kW: Int, kH: Int, pW: Int, pH: Int, sW: Int, sH: Int,
              dataFormat: TensorflowDataFormat, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("AvgPool")
      .putAttr("T", getDataType(value))
      .addInput(value.getName)
      .putAttr("data_format", dataFormat.value)
      .putAttr("ksize", kernelAttr(kW, kH, dataFormat))
      .putAttr("padding", getPaddingType(pW, pH, kW, kH, sW, sH).value)
      .putAttr("strides", strideAttr(sW, sH, dataFormat))
      .build()
  }

  def sigmoid(x: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Sigmoid")
      .putAttr("T", getDataType(x))
      .addInput(x.getName)
      .build()
  }

  def multiply(x: NodeDef, y: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Mul")
      .putAttr("T", getDataType(x))
      .addInput(x.getName)
      .addInput(y.getName)
      .build()
  }

  def floor(x: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Floor")
      .putAttr("T", getDataType(x))
      .addInput(x.getName)
      .build()
  }

  def add(x: NodeDef, y: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Add")
      .putAttr("T", getDataType(x))
      .addInput(x.getName)
      .addInput(y.getName)
      .build()
  }

  def realdiv(x: NodeDef, y: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("RealDiv")
      .putAttr("T", getDataType(x))
      .addInput(x.getName)
      .addInput(y.getName)
      .build()
  }

  def subtract(x: NodeDef, y: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Sub")
      .putAttr("T", getDataType(x))
      .addInput(x.getName)
      .addInput(y.getName)
      .build()
  }

  def shape(input: NodeDef, name: String, outType: DataType = DataType.DT_INT32): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Shape")
      .putAttr("T", getDataType(input))
      .putAttr("out_type", AttrValue.newBuilder().setType(outType).build())
      .build()
  }

  def randomUniform(shape: NodeDef, name: String, dtype: DataType = DataType.DT_FLOAT,
                    seed: Int = 0): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("RandomUniform")
      .putAttr("T", getDataType(shape))
      .putAttr("dtype", AttrValue.newBuilder().setType(dtype).build())
      .putAttr("seed", intAttr(seed))
      .putAttr("seed2", intAttr(seed))
      .addInput(shape.getName)
      .build()
  }

  def addN(inputs: Seq[NodeDef], name: String): NodeDef = {
    require(inputs.length >= 2, "at least two inputs for addN")
    val node = NodeDef.newBuilder()
      .setName(name)
      .putAttr("N", intAttr(inputs.length))
      .putAttr("T", getDataType(inputs(0)))
      .setOp("AddN")
    inputs.foreach(i => node.addInput(i.getName))
    node.build()
  }

  def concat(inputs: Seq[NodeDef], axis: Int, name: String): NodeDef = {
    require(inputs.length >= 1, "at least one inputs for addN")

    val node = NodeDef.newBuilder()
      .setName(name)
      .setOp("ConcatV2")
      .putAttr("N", intAttr(inputs.length - 1))
      .putAttr("T", getDataType(inputs(0)))
      .putAttr("Tidx", AttrValue.newBuilder().setType(DataType.DT_INT32).build())

    inputs.foreach(i => node.addInput(i.getName))

    node.build()
  }

  def pad(tensor: NodeDef, paddings: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Pad")
      .putAttr("T", getDataType(tensor))
      .putAttr("Tpaddings", getDataType(paddings))
      .addInput(tensor.getName)
      .addInput(paddings.getName)
      .build()
  }

  def reduceMean(inputTensor: NodeDef, axis: NodeDef, keepDim: Boolean, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Mean")
      .putAttr("T", getDataType(inputTensor))
      .putAttr("Tidx", getDataType(axis))
      .putAttr("keep_dims", booleanAttr(keepDim))
      .addInput(inputTensor.getName)
      .addInput(axis.getName)
      .build()
  }

  def softmax(logits: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Softmax")
      .putAttr("T", getDataType(logits))
      .addInput(logits.getName)
      .build()
  }

  def logSoftmax(logits: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("LogSoftmax")
      .putAttr("T", getDataType(logits))
      .addInput(logits.getName)
      .build()
  }

  def rsqrt(x: NodeDef, name: String): NodeDef = {
    NodeDef.newBuilder()
      .setName(name)
      .setOp("Rsqrt")
      .putAttr("T", getDataType(x))
      .addInput(x.getName)
      .build()
  }

  private def booleanAttr(value: Boolean): AttrValue = {
    AttrValue.newBuilder().setB(value).build()
  }

  private def intAttr(value: Int): AttrValue = {
    AttrValue.newBuilder().setI(value).build()
  }

  private def listIntAttr(value: Seq[Int]): AttrValue = {
    val list = ListValue.newBuilder()
    value.foreach(list.addI(_))
    AttrValue.newBuilder().setList(list).build()
  }

  private def tensorAttr[T: ClassTag](value: Tensor[T], dtype: DataType,
                                      byteOrder: ByteOrder, isScalar: Boolean): AttrValue = {
    val shape = TensorShapeProto.newBuilder()
    if (!isScalar) {
      value.size().foreach(dim => {
        shape.addDim(Dim.newBuilder().setSize(dim))
      })
    }
    require(value.isContiguous(), "only support save a contiguous tensor")

    val content = if (value.getType() == DoubleType) {
      val array = value.asInstanceOf[Tensor[Double]].storage().array()
      val offset = value.storageOffset() - 1
      if (dtype == DataType.DT_INT32) {
        val buffer = ByteBuffer.allocate(array.length * 4)
        buffer.order(byteOrder)
        var i = 0
        while (i < value.nElement()) {
          buffer.putInt(array(i + offset).toInt)
          i += 1
        }
        buffer
      } else if (dtype == DataType.DT_FLOAT) {
        val buffer = ByteBuffer.allocate(array.length * 4)
        buffer.order(byteOrder)
        var i = 0
        while (i < value.nElement()) {
          buffer.putFloat(array(i + offset).toFloat)
          i += 1
        }
        buffer
      } else if (dtype == DataType.DT_DOUBLE) {
        val buffer = ByteBuffer.allocate(array.length * 8)
        buffer.order(byteOrder)
        var i = 0
        while (i < value.nElement()) {
          buffer.putDouble(array(i + offset))
          i += 1
        }
        buffer
      } else {
        throw new UnsupportedOperationException(s"data type ${dtype} is not supported currently")
      }
    } else {
      val array = value.asInstanceOf[Tensor[Float]].storage().array()
      val offset = value.storageOffset() - 1
      if (dtype == DataType.DT_INT32) {
        val buffer = ByteBuffer.allocate(array.length * 4)
        buffer.order(byteOrder)
        var i = 0
        while (i < value.nElement()) {
          buffer.putInt(array(i + offset).toInt)
          i += 1
        }
        buffer
      } else if (dtype == DataType.DT_FLOAT) {
        val buffer = ByteBuffer.allocate(array.length * 4)
        buffer.order(byteOrder)
        var i = 0
        while (i < value.nElement()) {
          buffer.putFloat(array(i + offset))
          i += 1
        }
        buffer
      } else if (dtype == DataType.DT_DOUBLE) {
        throw new IllegalArgumentException(s"can not convert a float tensor to double tensor")
      } else {
        throw new UnsupportedOperationException(s"data type ${dtype} is not supported currently")
      }
    }

    AttrValue.newBuilder().setTensor(
      TensorProto.newBuilder().setTensorShape(shape).setDtype(dtype)
        .setTensorContent(ByteString.copyFrom(content.array()))
    ).build()
  }

  private def tensorAttr(value: Seq[Int]): AttrValue = {
    val shape = TensorShapeProto.newBuilder()
    shape.addDim(Dim.newBuilder().setSize(value.length))
    val dtype = DataType.DT_INT32
    AttrValue.newBuilder().setTensor(
      TensorProto.newBuilder().setTensorShape(shape).setDtype(dtype)
    ).build()
  }

  private def typeAttr(dtype : TensorDataType): AttrValue = {
    if (dtype == FloatType) {
      AttrValue.newBuilder().setType(DataType.DT_FLOAT).build()
    } else if (dtype == DoubleType) {
      AttrValue.newBuilder().setType(DataType.DT_DOUBLE).build()
    } else {
      throw new NotImplementedError(s"type $dtype is not supported")
    }
  }

  private def shapeAttr(shape: Seq[Int]): AttrValue = {
    val attr = TensorShapeProto.newBuilder()
    shape.foreach(dim => {
      attr.addDim(Dim.newBuilder().setSize(dim))
    })
    AttrValue.newBuilder().setShape(attr).build()
  }

  private def getDataType(node: NodeDef): AttrValue = {
    var attr = node.getAttrOrDefault("dtype", null)
    if (attr != null) {
      return attr
    }

    attr = node.getAttrOrDefault("out_type", null)
    if (attr != null) {
      return attr
    }

    attr = node.getAttrOrDefault("T", null)
    if (attr != null) {
      return attr
    }

    throw new IllegalArgumentException("TensorflowSaver: Can not find data type")
  }

  private def getPaddingType(padW: Int, padH: Int, kW: Int, kH: Int, sW: Int, sH: Int)
      : PaddingType = {
    if (padW == 0 && padH == 0) {
      return PaddingType.PADDING_VALID
    } else if (2 * padW == (kW - sW) && 2 * padH == (kH - sH)) {
      return PaddingType.PADDING_SAME
    } else {
      throw new IllegalArgumentException(
        s"Can not get padding type from given parameter " +
          s"(padW: $padW, padH: $padH, kW: $kW, kH: $kH, sW: $sW, sH: $sH )")
    }
  }

  private def kernelAttr(kW: Int, kH: Int, dataFormat: TensorflowDataFormat): AttrValue = {
    val kSize = if (dataFormat == TensorflowDataFormat.NHWC) {
      Seq(1, kH, kW, 1)
    } else {
      Seq(1, 1, kH, kW)
    }
    listIntAttr(kSize)
  }

  private def strideAttr(sW: Int, sH: Int, dataFormat: TensorflowDataFormat): AttrValue = {
    val sSize = if (dataFormat == TensorflowDataFormat.NHWC) {
      Seq(1, sH, sW, 1)
    } else {
      Seq(1, 1, sH, sW)
    }
    listIntAttr(sSize)
  }
}
