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

import java.io.FileOutputStream

import com.google.protobuf.CodedOutputStream
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger
import org.tensorflow.framework._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import TensorflowUtils._

object TensorFlowSaver {
  /**
   * Save a graph model to protobuf files so that it can be used in tensorflow inference.
   *
   * When save the model, placeholders will be added to the tf model as input nodes. So you need to
   * pass in the names and shape for the placeholders. BigDL model doesn't have such information.
   * The order of the placeholde information should be same as the inputs of the graph model
   *
   * @param model graph model instance
   * @param inputs placeholder information
   * @param path where to save
   * @tparam T
   */
  def saveGraph[T](
      model : Graph[T],
      inputs : Seq[(String, Seq[Int])],
      path: String): Unit = {
    val inputNodeDefs = inputs.map(input =>
      placeholder(model.getNumericType(), input._2, input._1)
    )
    val inputNodeCache =
      new mutable.HashMap[AbstractModule[Activity, Tensor[T], T], ArrayBuffer[NodeDef]]()
    model.inputs.zip(inputNodeDefs).foreach(n => {
      inputNodeCache(n._1.element) = ArrayBuffer(n._2)
    })

    val graphBuilder = GraphDef.newBuilder()
    inputNodeDefs.foreach(graphBuilder.addNode(_))

    model.executions.foreach(n => {
      val nodeDefs = maps(n.element.getClass.getName).toTFDef(n.element, inputNodeCache(n.element))
      nodeDefs.foreach(nDef => {
        graphBuilder.addNode(nDef)
      })
      n.nextNodes.foreach(n => {
        val list = inputNodeCache.getOrElse(n.element, ArrayBuffer())
        list.append(nodeDefs(0))
      })
    })

    // Save to file
    val os = new FileOutputStream(path)
    val output = CodedOutputStream.newInstance(os)
    val graph = graphBuilder.build()
    logger.debug("Graph definition is:")
    logger.debug(graph.toString)
    graph.writeTo(output)
    output.flush()
    os.close()
    logger.info(s"Save as tensorflow model file to $path")
  }

  private val logger = Logger.getLogger(getClass)

  private val maps = mutable.Map[String, BigDLToTF](
    getNameFromObj(ReLU.getClass.getName) -> ReLUToTF,
    getNameFromObj(Linear.getClass.getName) -> LinearToTF
  )

  private def getNameFromObj(name: String) : String = name.substring(0, name.length - 1)
}

/**
 * Wrapper of logic to convert module to tensorflow node definition
 */
trait BigDLToTF {

  /**
   * Convert the module to a tensorflow nodedef
   * @return Mapped nodedef list, the first is the output node
   */
  def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef]
}

object ReLUToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Relu only accept one input")

    Seq(relu(inputs(0), module.getName()))
  }
}

object LinearToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Linear only accept one input")
    val linear = module.asInstanceOf[Linear[_]]
    val weight = const(linear.weight, linear.getName() + "/weight")
    val weightReader = identity(weight, linear.getName() + "/weightReader")
    val mm = matmul(inputs(0), weightReader, linear.getName() + "matmul")
    val bias = const(linear.bias, linear.getName() + "/bias")
    val biasReader = identity(bias, linear.getName() + "/biasReader")
    val add = biasAdd(mm, biasReader, NCHW, linear.getName() + "/biasAdd")
    Seq(add, biasReader, bias, mm, weightReader, weight)
  }
}

object SpatialConvolutionToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "SpatialConvolution only accept one input")
    val spatialConv = module.asInstanceOf[SpatialConvolution[_]]
    val filter = const(spatialConv.weight, spatialConv.getName() + "/filter")
    val filterReader = identity(filter, spatialConv.getName() + "/filterReader")
    val conv = conv2D(inputs(0), filterReader, spatialConv.strideH, spatialConv.strideW,
      spatialConv.kernelW, spatialConv.kernelH, spatialConv.strideW, spatialConv.strideH, NCHW,
      spatialConv.getName() + "/conv2D")
    val bias = const(spatialConv.bias, spatialConv.getName() + "/bias")
    val biasReader = identity(bias, spatialConv.getName() + "/biasReader")
    val add = biasAdd(conv, biasReader, NCHW, spatialConv.getName() + "/biasAdd")
    Seq(add, biasReader, bias, conv, filterReader, filter)
  }
}

object SqueezeToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Squeeze only accept one input")
    val sq = module.asInstanceOf[Squeeze[_]]
    Seq(squeeze(inputs(0), sq.dims.map(_ - 1), sq.getName()))
  }
}

object TanhToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Tanh only accept one input")
    Seq(tanh(inputs(0), module.getName()))
  }
}

object ReshapeToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Reshape only accept one input")
    val rh = module.asInstanceOf[Reshape[_]]
    val size = Tensor[Float](rh.size.length)
    var i = 0
    while(i < rh.size.length) {
      size.setValue(i + 1, rh.size(i))
      i += 1
    }
    val shape = const(size, rh.getName() + "/shape", DataType.DT_INT32)
    val reshapeNode = reshape(inputs(0), shape, rh.getName())
    Seq(reshapeNode, shape)
  }
}

object ViewToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Reshape only accept one input")
    val viewLayer = module.asInstanceOf[View[_]]
    val size = Tensor[Float](viewLayer.sizes.length)
    var i = 0
    while(i < viewLayer.sizes.length) {
      size.setValue(i + 1, viewLayer.sizes(i))
      i += 1
    }
    val shape = const(size, viewLayer.getName() + "/shape", DataType.DT_INT32)
    val reshapeNode = reshape(inputs(0), shape, viewLayer.getName())
    Seq(reshapeNode, shape)
  }
}

object MaxpoolToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Maxpool only accept one input")
    val layer = module.asInstanceOf[SpatialMaxPooling[_]]
    Seq(maxPool(inputs(0), layer.kW, layer.kH, layer.padW, layer.padH,
      layer.dW, layer.dH, NCHW, layer.getName()))
  }
}

object PaddingToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Padding only accept one input")
    val layer = module.asInstanceOf[Padding[_]]
    val padding = Tensor[Float](1, 2)
    if (layer.pad < 0) {
      padding.setValue(1, 1, -layer.pad)
      padding.setValue(1, 2, 0)
    }
    else {
      padding.setValue(1, 1, 0)
      padding.setValue(1, 2, layer.pad)
    }
    val paddingsNode = const(padding, layer.getName() + "/padding", DataType.DT_INT32)
    val padNode = pad(inputs(0), paddingsNode, layer.getName() + "/output")
    Seq(padNode, paddingsNode)
  }
}

object AvgpoolToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Avgpool only accept one input")
    val layer = module.asInstanceOf[SpatialAveragePooling[_]]
    Seq(avgPool(inputs(0), layer.kW, layer.kH, layer.padW, layer.padH,
      layer.dW, layer.dH, NCHW, layer.getName()))
  }
}

object SigmoidToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Sigmoid only accept one input")
    Seq(sigmoid(inputs(0), module.getName()))
  }
}

object DropoutToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Dropout only accept one input")
    val layer = module.asInstanceOf[Dropout[_]]
    val shapeNode = shape(inputs(0), layer.getName() + "/shape")
    val rand = randomUniform(shapeNode, layer.getName() + "/random")
    val maxNode = const(Tensor[Float](T(1.0f)), layer.getName() + "/max")
    val minNode = const(Tensor[Float](T(0.0f)), layer.getName() + "/max")
    val sub = subtract(maxNode, minNode, layer.getName() + "/sub")
    val mul = multiply(rand, sub, layer.getName() + "/mul")
    val randOutput = add(minNode, mul, layer.getName() + "/rand_output")
    val keepProb = const(Tensor[Float](T(0.5f)), layer.getName() + "/keep_prob")
    val div1 = realdiv(keepProb, inputs(0), layer.getName() + "/div1")
    val div2 = realdiv(keepProb, randOutput, layer.getName() + "/div2")
    val floorNode = floor(div2, layer.getName() + "/floor")
    val output = multiply(div1, floorNode, layer.getName() + "/output")
    Seq(output, floorNode, div2, div1, keepProb, randOutput, mul, sub, minNode, maxNode,
      rand, shapeNode)
  }
}

object CAddTableToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    Seq(addN(inputs, module.getName()))
  }
}

object CMultTableToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 2, "Tensorflow only support two tensor multiply together")

    Seq(multiply(inputs(0), inputs(1), module.getName()))
  }
}

object JoinTableToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    val layer = module.asInstanceOf[JoinTable[_]]
    Seq(concat(inputs, layer.dimension - 1, layer.getName()))
  }
}

object MeanToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Mean only accept one input")
    val layer = module.asInstanceOf[Mean[_]]
    val dimsTensor = Tensor[Float](layer.dimension)
    dimsTensor.setValue(1, layer.dimension)

    val dims = const(dimsTensor, layer.getName() + "/dims")
    val mean = reduceMean(inputs(0), dims, false, layer.getName() + "/output")
    Seq(mean, dims)
  }
}

object SoftMaxToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "Softmax only accept one input")
    Seq(softmax(inputs(0), module.getName()))
  }
}

object LogSoftMaxToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "LogSoftmax only accept one input")
    Seq(logSoftmax(inputs(0), module.getName()))
  }
}

object BatchNormToTF extends BigDLToTF {
  override def toTFDef(module: AbstractModule[_, _, _], inputs: Seq[NodeDef]): Seq[NodeDef] = {
    require(inputs.length == 1, "BatchNorm only accept one input")
    val layer = module.asInstanceOf[SpatialBatchNormalization[_]]
    val stdVar = const(layer.saveStd, layer.getName() + "/std")
    val mean = const(layer.saveMean, layer.getName() + "/mean")
    val scale = const(layer.weight, layer.getName() + "/scale")
    val offset = const(layer.bias, layer.getName() + "/offset")
    val div = realdiv(scale, stdVar, layer.getName() + "/div")
    val mul1 = multiply(inputs(0), div, layer.getName() + "/mul1")
    val mul2 = multiply(scale, div, layer.getName() + "/mul2")
    val sub = multiply(offset, scale, layer.getName() + "/sub")
    val output = add(mul1, sub, layer.getName() + "/output")
    Seq(output, sub, mul2, mul1, div, offset, scale, mean, stdVar)
  }
}