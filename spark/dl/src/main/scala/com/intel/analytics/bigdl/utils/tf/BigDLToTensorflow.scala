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

import java.nio.ByteOrder

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import Tensorflow._
import BigDLToTensorflow._
import org.tensorflow.framework.{DataType, NodeDef}

import scala.collection.mutable.ArrayBuffer

/**
 * Wrapper of logic to convert module to tensorflow node definition
 */
trait BigDLToTensorflow {

  /**
   * Convert the module to a tensorflow nodedef
   * @return Mapped nodedef list, the first is the output node
   */
  def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
              byteOrder: ByteOrder): Seq[NodeDef]
}

object BigDLToTensorflow {
  /**
   * This method is just for test purpose. Do not use the bigdl.saveNHWC for real use case
   * @return
   */
  private[tf] def processSaveDim(dim: Int): Int = {
    if (System.getProperty("bigdl.enableNHWC", "false").toBoolean) {
      if (dim == 2) return 4
      if (dim == 3) return 2
      if (dim == 4) return 3
      dim
    } else {
      dim
    }
  }

  /**
   * This method is just for test purpose. Do not use the bigdl.enableNHWC for real use case
   * @return
   */
  private[tf] def getDataFormat(): TensorflowDataFormat = {
    if (System.getProperty("bigdl.enableNHWC", "false").toBoolean) {
      TensorflowDataFormat.NHWC
    } else {
      TensorflowDataFormat.NCHW
    }
  }
}

object InputToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Input only accept one input")

    Seq(identity(inputs(0), module.getName()))
  }
}

object ReLUToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Relu only accept one input")

    Seq(relu(inputs(0), module.getName()))
  }
}

object LinearToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Linear only accept one input")
    val linear = module.asInstanceOf[Linear[_]]
    val weight = const(linear.weight.t().contiguous(), linear.getName() + "/weight", byteOrder)
    val weightReader = identity(weight, linear.getName() + "/weightReader")
    val mm = matmul(inputs(0), weightReader, linear.getName() + "/matmul")
    val bias = const(linear.bias, linear.getName() + "/bias", byteOrder)
    val biasReader = identity(bias, linear.getName() + "/biasReader")
    val addNode = add(mm, biasReader, linear.getName() + "/add")
    Seq(addNode, biasReader, bias, mm, weightReader, weight)
  }
}

object SpatialConvolutionToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    import scala.language.existentials
    require(inputs.length == 1, "SpatialConvolution only accept one input")
    val spatialConv = module.asInstanceOf[SpatialConvolution[_]]
    if (spatialConv.nGroup == 1) {
      val (dataFormat, filterTensor) = if (spatialConv.format == DataFormat.NCHW) {
        (TensorflowDataFormat.NCHW,
          spatialConv.weight.select(1, 1)
            .transpose(2, 3).transpose(3, 4)
            .transpose(1, 2).transpose(2, 3)
            .transpose(3, 4).contiguous())
      } else {
        (TensorflowDataFormat.NHWC, spatialConv.weight.select(1, 1))
      }

      val filter = const(filterTensor, spatialConv.getName() + "/filter", byteOrder)
      val filterReader = identity(filter, spatialConv.getName() + "/filterReader")
      val conv = conv2D(inputs(0), filterReader, spatialConv.strideW, spatialConv.strideH,
        spatialConv.kernelW, spatialConv.kernelH, spatialConv.padW, spatialConv.padH,
        dataFormat, spatialConv.getName() + "/conv2D")
      if (spatialConv.bias != null) {
        val bias = const(spatialConv.bias, spatialConv.getName() + "/bias", byteOrder)
        val biasReader = identity(bias, spatialConv.getName() + "/biasReader")
        val add = biasAdd(conv, biasReader, dataFormat,
          spatialConv.getName() + "/biasAdd")
        Seq(add, biasReader, bias, conv, filterReader, filter)
      } else {
        Seq(conv, filterReader, filter)
      }
    } else {
      require(spatialConv.format == DataFormat.NCHW, "Only NCHW support conv group")
      val nodes = new ArrayBuffer[NodeDef]()
      val splitDim = const(Tensor.scalar[Int](1), spatialConv.getName() + "/split_dim",
        ByteOrder.LITTLE_ENDIAN)
      val splits = split(splitDim, inputs(0), spatialConv.nGroup, spatialConv.getName() + "/split")
      nodes.append(splitDim)
      nodes.appendAll(splits)
      val axis = const(Tensor.scalar[Int](1), spatialConv.getName() + "/concat/axis",
        ByteOrder.LITTLE_ENDIAN)
      nodes.append(axis)
      val outputs = (0 until spatialConv.nGroup).map(g => {
        val filterTensor = spatialConv.weight.select(1, g + 1)
          .transpose(2, 3).transpose(3, 4)
          .transpose(1, 2).transpose(2, 3)
          .transpose(3, 4).contiguous()

        val filter = const(filterTensor, spatialConv.getName() + s"/group$g/filter", byteOrder)
        val filterReader = identity(filter, spatialConv.getName() + s"/group$g/filterReader")
        val conv = conv2D(splits(g), filterReader, spatialConv.strideW, spatialConv.strideH,
          spatialConv.kernelW, spatialConv.kernelH, spatialConv.padW, spatialConv.padH,
          TensorflowDataFormat.NCHW, spatialConv.getName() + s"/group$g/conv2D")
        if (spatialConv.bias != null) {
          val bias = const(spatialConv.bias.narrow(1,
            g * spatialConv.nOutputPlane / spatialConv.nGroup + 1,
            spatialConv.nOutputPlane / spatialConv.nGroup),
            spatialConv.getName() + s"/group$g/bias", byteOrder)
          val biasReader = identity(bias, spatialConv.getName() + s"/group$g/biasReader")
          val add = biasAdd(conv, biasReader, TensorflowDataFormat.NCHW,
            spatialConv.getName() + s"/group$g/biasAdd")
          nodes.append(add, biasReader, bias, conv, filterReader, filter)
          add
        } else {
          nodes.append(conv, filterReader, filter)
          conv
        }
      }) ++ Seq(axis)

      val concatNode = concat(outputs, spatialConv.getName() + "/concat/output")
      Seq(concatNode) ++ nodes
    }
  }
}

object TemporalConvolutionToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "SpatialConvolution only accept one input")
    val spatialConv = module.asInstanceOf[TemporalConvolution[_]]

    val const1 = const(Tensor.scalar[Int](1), spatialConv.getName() + "/dim1", byteOrder)
    val expandDimsInput = expandDims(inputs.head, const1,
      spatialConv.getName() + "/expandDimsInput")

    val filterTensor = spatialConv.weight
      .view(spatialConv.outputFrameSize, spatialConv.kernelW, spatialConv.inputFrameSize)
      .transpose(2, 3).transpose(1, 3).contiguous()

    val filter = const(filterTensor, spatialConv.getName() + "/filter", byteOrder)
    val filterReader = identity(filter, spatialConv.getName() + "/filterReader")

    val const2 = const(Tensor.scalar[Int](0), spatialConv.getName() + "/dim2", byteOrder)
    val expandDimsWeight = expandDims(filterReader, const2,
      spatialConv.getName() + "/expandDimsWeight")

    val conv = conv2D(expandDimsInput, expandDimsWeight, spatialConv.strideW, 1,
      spatialConv.kernelW, 1, 0, 0,
      getDataFormat(), spatialConv.getName() + "/conv2D")

    val sq = squeeze(conv, Seq(1), spatialConv.getName() + "/squeeze")

    val bias = const(spatialConv.bias, spatialConv.getName() + "/bias", byteOrder)
    val biasReader = identity(bias, spatialConv.getName() + "/biasReader")
    val add = biasAdd(sq, biasReader, getDataFormat(),
      spatialConv.getName() + "/biasAdd")
    Seq(add, biasReader, bias, conv, filterReader, filter, sq,
      expandDimsInput, expandDimsWeight, const1, const2)
  }
}

object SqueezeToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Squeeze only accept one input")
    val sq = module.asInstanceOf[Squeeze[_]]
    Seq(squeeze(inputs(0), sq.dims.map(processSaveDim(_) - 1), sq.getName()))
  }
}

object TanhToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Tanh only accept one input")
    Seq(tanh(inputs(0), module.getName()))
  }
}

object ReshapeToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Reshape only accept one input")
    val rh = module.asInstanceOf[Reshape[_]]
    val size = Tensor[Int](rh.size.length)
    var i = 0
    while(i < rh.size.length) {
      size.setValue(i + 1, rh.size(i))
      i += 1
    }
    val shape = const(size, rh.getName() + "/shape", byteOrder)
    val reshapeNode = reshape(inputs(0), shape, rh.getName())
    Seq(reshapeNode, shape)
  }
}

object ViewToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Reshape only accept one input")
    val viewLayer = module.asInstanceOf[View[_]]
    val size = Tensor[Int](viewLayer.sizes.length + 1).setValue(1, -1)
    var i = 1
    while(i < viewLayer.sizes.length + 1) {
      size.setValue(i + 1, viewLayer.sizes(i - 1))
      i += 1
    }

    val shape = const(size, viewLayer.getName() + "/shape", byteOrder)
    val reshapeNode = reshape(inputs(0), shape, viewLayer.getName())
    Seq(reshapeNode, shape)
  }
}

object MaxpoolToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Maxpool only accept one input")
    val layer = module.asInstanceOf[SpatialMaxPooling[_]]
    val dataFormat = if (layer.format == DataFormat.NHWC) {
      TensorflowDataFormat.NHWC
    } else {
      TensorflowDataFormat.NCHW
    }
    Seq(maxPool(inputs(0), layer.kW, layer.kH, layer.padW, layer.padH,
      layer.dW, layer.dH, dataFormat, layer.getName(), layer.ceilMode))
  }
}

object PaddingToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Padding only accept one input")
    val layer = module.asInstanceOf[Padding[_]]
    require(layer.nIndex == 1, "only support padding nIndex == 1")
    require(layer.nInputDim > 0, "nInputDim must be explicit specified")
    val padding = Tensor[Int](layer.nInputDim, 2).zero()
    if (layer.pad < 0) {
      padding.setValue(layer.dim, 1, -layer.pad)
    }
    else {
      padding.setValue(layer.dim, 2, layer.pad)
    }
    val paddingsNode = const(padding, layer.getName() + "/padding", byteOrder)
    val padNode = pad(inputs(0), paddingsNode, layer.getName() + "/output")
    Seq(padNode, paddingsNode)
  }
}

object AvgpoolToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Avgpool only accept one input")
    val layer = module.asInstanceOf[SpatialAveragePooling[_]]
    val dataFormat = if (layer.format == DataFormat.NHWC) {
      TensorflowDataFormat.NHWC
    } else {
      TensorflowDataFormat.NCHW
    }
    Seq(avgPool(inputs(0), layer.kW, layer.kH, layer.padW, layer.padH,
      layer.dW, layer.dH, dataFormat, layer.getName(), layer.ceilMode))
  }
}

object SigmoidToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Sigmoid only accept one input")
    Seq(sigmoid(inputs(0), module.getName()))
  }
}

object DropoutToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Dropout only accept one input")
    val layer = module.asInstanceOf[Dropout[_]]
    require(layer.isTraining() == false, "only support evaluating mode dropout")
    require(inputs.length == 1, "require only one tensor input")
    Seq(identity(inputs(0), layer.getName()))
  }
}

object ScaleToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
    byteOrder: ByteOrder): Seq[NodeDef] = {
    val layer = module.asInstanceOf[Scale[_]]
    val weight = const(layer.cmul.weight, layer.getName() + "/mul/weight", ByteOrder.LITTLE_ENDIAN)
    val mulNode = multiply(weight, inputs(0), layer.getName() + "/mul/mul")
    val bias = const(layer.cadd.bias, layer.getName() + "/add/bias", ByteOrder.LITTLE_ENDIAN)
    val output = add(mulNode, bias, layer.getName() + "/add/add")
    Seq(output, bias, mulNode, weight)
  }
}

object CAddTableToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    Seq(addN(inputs, module.getName()))
  }
}

object CMultTableToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 2, "Tensorflow only support two tensor multiply together")

    Seq(multiply(inputs(0), inputs(1), module.getName()))
  }
}

object JoinTableToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    val layer = module.asInstanceOf[JoinTable[_]]
    val axis = const(Tensor.scalar[Int](layer.dimension - 1), layer.getName() + "/axis", byteOrder)
    val updateInputs = new ArrayBuffer[NodeDef]()
    updateInputs ++= inputs.reverse
    updateInputs.append(axis)
    Seq(concat(updateInputs, layer.getName()), axis)
  }
}

object MeanToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Mean only accept one input")
    val layer = module.asInstanceOf[Mean[_]]
    require(layer.squeeze == true, "Mean must squeeze input")
    val dimsTensor = Tensor[Int](layer.dimension)
    dimsTensor.setValue(1, layer.dimension - 1)

    val dims = const(dimsTensor, layer.getName() + "/dims", byteOrder)
    val mean = reduceMean(inputs(0), dims, false, layer.getName() + "/output")
    Seq(mean, dims)
  }
}

object SoftMaxToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "Softmax only accept one input")
    Seq(softmax(inputs(0), module.getName()))
  }
}

object LogSoftMaxToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "LogSoftmax only accept one input")
    Seq(logSoftmax(inputs(0), module.getName()))
  }
}

object BatchNorm2DToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
                       byteOrder: ByteOrder): Seq[NodeDef] = {
    require(inputs.length == 1, "BatchNorm only accept one input")
    val layer = module.asInstanceOf[SpatialBatchNormalization[_]]
    require(!layer.isTraining(), "Only support evaluate mode batch norm")
    // reshape to nchw
    val size = Tensor[Int](layer.nDim)
    for (i <- 0 until layer.nDim) {
      size.setValue(i + 1, 1)
    }

    size(2) = layer.runningVar.size(1)
    if (layer.weight != null) {
      val shapeVar = const(size, layer.getName() + "/reshape_1/shape", byteOrder)
      val shapeMean = const(size, layer.getName() + "/reshape_2/shape", byteOrder)
      val shapeScale = const(size, layer.getName() + "/reshape_3/shape", byteOrder)
      val shapeOffset = const(size, layer.getName() + "/reshape_4/shape", byteOrder)

      val varNode = const(layer.runningVar, layer.getName() + "/var", byteOrder)
      val mean = const(layer.runningMean, layer.getName() + "/mean", byteOrder)
      val scale = const(layer.weight, layer.getName() + "/scale", byteOrder)
      val offset = const(layer.bias, layer.getName() + "/offset", byteOrder)
      val reshapeVar = reshape(varNode, shapeVar, s"${layer.getName()}/reshape_1")
      val reshapeMean = reshape(mean, shapeMean, s"${layer.getName()}/reshape_2")
      val reshapeScale = reshape(scale, shapeScale, s"${layer.getName()}/reshape_3")
      val reshapeOffset = reshape(offset, shapeOffset, s"${layer.getName()}/reshape_4")
      // construct graph
      val sqrtVar = rsqrt(reshapeVar, layer.getName() + "/sqrtvar")
      val mul0 = multiply(reshapeScale, sqrtVar, layer.getName() + "/mul0")
      val mul1 = multiply(inputs(0), mul0, layer.getName() + "/mul1")
      val mul2 = multiply(reshapeMean, mul0, layer.getName() + "/mul2")
      val sub = subtract(reshapeOffset, mul2, layer.getName() + "/sub")
      val output = add(mul1, sub, layer.getName() + "/output")
      Seq(output, sub, mul2, mul1, mul0, reshapeOffset, reshapeMean, reshapeScale,
        shapeOffset, shapeMean, shapeScale, offset, scale, mean,
        sqrtVar, reshapeVar, shapeVar, varNode)
    } else {
      val shapeVar = const(size, layer.getName() + "/reshape_1/shape", byteOrder)
      val shapeMean = const(size, layer.getName() + "/reshape_2/shape", byteOrder)

      val varNode = const(layer.runningVar, layer.getName() + "/var", byteOrder)
      val mean = const(layer.runningMean, layer.getName() + "/mean", byteOrder)
      val reshapeVar = reshape(varNode, shapeVar, s"${layer.getName()}/reshape_1")
      val reshapeMean = reshape(mean, shapeMean, s"${layer.getName()}/reshape_2")
      // construct graph
      val sqrtVar = rsqrt(reshapeVar, layer.getName() + "/sqrtvar")
      val mul1 = multiply(inputs(0), sqrtVar, layer.getName() + "/mul1")
      val mul2 = multiply(reshapeMean, sqrtVar, layer.getName() + "/mul2")
      val output = subtract(mul1, mul2, layer.getName() + "/output")
      Seq(output, mul2, mul1, reshapeMean, shapeMean, mean, sqrtVar, reshapeVar, shapeVar, varNode)
    }
  }
}

object LRNToTF extends BigDLToTensorflow {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef],
    byteOrder: ByteOrder): Seq[NodeDef] = {
    val layer = module.asInstanceOf[SpatialCrossMapLRN[_]]
    if (layer.format == DataFormat.NHWC) {
      Seq(lrn(inputs(0), (layer.size - 1) / 2, layer.k.toFloat, (layer.alpha / layer.size).toFloat,
        layer.beta.toFloat, module.getName()))
    } else {
      val perm1 = const(Tensor[Int](T(0, 2, 3, 1)), module.getName() + "/perm1",
        ByteOrder.LITTLE_ENDIAN)
      val transpose1 = transpose(inputs(0), perm1, module.getName() + "/transpose1")
      val lrnNode = lrn(transpose1, (layer.size - 1) / 2, layer.k.toFloat,
        (layer.alpha / layer.size).toFloat,
        layer.beta.toFloat, module.getName() + "/lrn")
      val perm2 = const(Tensor[Int](T(0, 3, 1, 2)), module.getName() + "/perm2",
        ByteOrder.LITTLE_ENDIAN)
      val output = transpose(lrnNode, perm2, module.getName() + "/transpose2")
      Seq(output, perm1, transpose1, lrnNode, perm2)
    }
  }
}
