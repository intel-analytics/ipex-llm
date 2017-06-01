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

import java.nio.charset.Charset
import java.nio.{ByteBuffer, ByteOrder}

import collection.JavaConverters._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.tensorflow.framework.{DataType, NodeDef, TensorProto}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}

import scala.collection.mutable.ArrayBuffer

trait TFToBigDL {
  def topology: DirectedGraph[String]

  def layer(tfGraph: DirectedGraph[NodeDef],
            context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float])


  /**
   * @param tfGraph
   * @return the nodes of this subgraph that accept input data
   */
  def getInputNodes(tfGraph: DirectedGraph[NodeDef]): Seq[Node[NodeDef]] = null

  /**
   *
   * @param tfGraph
   * @return the nodes of this subgraph that emit output data
   */
  def getOutputNodes(tfGraph: DirectedGraph[NodeDef]): Seq[Node[NodeDef]] = {
    Seq(tfGraph.source)
  }

  protected def getOrSetTensor(node: NodeDef, context: Context)
                              (f: Tensor[Float] => Tensor[Float])
  : (Tensor[Float], Tensor[Float]) = {
    if (context.contains(node)) {
      context(node)
    } else {
      val weight = f(TFToBigDL.toTensor(node.getAttrMap.get("value").getTensor)).contiguous()
      val gradient = Tensor[Float](weight.size())
      context.put(node, (weight, gradient))
      (weight, gradient)
    }
  }
}

object FullConnectionTF extends TFToBigDL{
  private val graph = {
    val add = Node("BiasAdd")
    val mul = Node("MatMul")
    Node("*") -> mul
    Node("Const") -> Node("Identity") -> mul -> add
    Node("Const") -> Node("Identity") -> add
    add.graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : (AbstractModule[Activity, Tensor[Float], Float]) = {

    val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.element
    val weightNode = tfGraph.source.prevNodes.head.prevNodes(1).prevNodes.head.element
    val (bias, gradBias) = getOrSetTensor(biasNode, context)(t => t)
    val (weight, gradWeight) = getOrSetTensor(weightNode, context) { t =>
      t.transpose(1, 2)
    }

    val linearLayer = Linear[Float](inputSize = weight.size(2), outputSize = weight.size(1),
      initWeight = weight, initGradWeight = gradWeight, initBias = bias, initGradBias = gradBias)
    linearLayer.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }

  override def getInputNodes(tfGraph: DirectedGraph[NodeDef]): Seq[Node[NodeDef]] = {
    Seq(tfGraph.source.prevNodes.head)
  }
}

object  SqueezeTF extends TFToBigDL {
  private val graph = (Node("*") -> Node("Squeeze")).graph(reverse = true)
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
    : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val dataFormatMatch = Map("N" -> 0, "H" -> 2, "W" -> 3, "C" -> 1)
    val dims = tfGraph.source.element.getAttrOrThrow("squeeze_dims").getList().getIList()
      .asScala.map(_.toInt).toArray
      .map(i => dataFormatMatch(TFToBigDL.dataFormat.charAt(i).toString))
    Squeeze[Float](dims, batchMode = true)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object Conv2D extends TFToBigDL{
  private val graph = {
    val add = Node("BiasAdd")
    val conv = Node("Conv2D")

    Node("*") -> conv
    Node("Const") -> Node("Identity") -> conv -> add
    Node("Const") -> Node("Identity") -> add
    add.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def getInputNodes(tfGraph: DirectedGraph[NodeDef]): Seq[Node[NodeDef]] = {
    Seq(tfGraph.source.prevNodes.head)
  }

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val attributes = tfGraph.source.prevNodes(0).element.getAttrMap
    require(attributes.get("strides").getList.getI(0).toInt == 1, s"not support strides on batch")

    val (strideH, strideW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(attributes.get("strides").getList.getI(3).toInt == 1, s"not support strides on depth")
      (attributes.get("strides").getList.getI(1).toInt,
        attributes.get("strides").getList.getI(2).toInt)
    } else if (attributes.get("data_format").getS.toString(Charset.defaultCharset()) == "NCHW") {
      require(attributes.get("strides").getList.getI(2).toInt == 1, s"not support strides on depth")
      (attributes.get("strides").getList.getI(2).toInt,
        attributes.get("strides").getList.getI(3).toInt)
    } else {
      throw new IllegalArgumentException("no supported data format")
    }
    val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.element
    val (bias, gradBias) = getOrSetTensor(biasNode, context)(t => t)

    val weightNode = tfGraph.source.prevNodes.head.prevNodes(1).prevNodes.head.element
    val (weights, gradWeights) = getOrSetTensor(weightNode, context) { t =>
      t.transpose(1, 4).transpose(2, 3).transpose(3, 4)
    }

    val nOuputPlane = weights.size(1)
    val nInputPlane = weights.size(2)
    val kernelH = weights.size(3)
    val kernelW = weights.size(4)

    val (pW, pH) =
      if (attributes.get("padding").getS.toString(Charset.defaultCharset()) == "SAME") {
        require((kernelW - strideW) % 2 == 0)
        require((kernelH - strideH) % 2 == 0)
        ((kernelW - strideW) / 2, (kernelH - strideH) / 2)
      } else {
        (0, 0)
      }

    val convLayer = SpatialConvolution[Float](
      nInputPlane = nInputPlane, nOutputPlane = nOuputPlane,
      kernelW = kernelW, kernelH = kernelH,
      strideW = strideW, strideH = strideH,
      padW = pW, padH = pH,
      initWeight = weights,
      initBias = bias,
      initGradWeight = gradWeights,
      initGradBias = gradBias
    )
    convLayer.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ReluTF extends  TFToBigDL {
  private val graph = {
    (Node("*") -> Node("Relu")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    ReLU[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object TanhTF extends  TFToBigDL{
  private val graph = {
    (Node("*") -> Node("Tanh")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    Tanh[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object SigmoidTF extends  TFToBigDL{
  private val graph = {
    (Node("*") -> Node("Sigmoid")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    Sigmoid[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ReshapeTF extends TFToBigDL {
  private val graph = {
    val nodeReshape = Node("Reshape")
    Node("*") -> nodeReshape
    Node("Const") -> nodeReshape
    nodeReshape.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val sizes = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(1).element.getAttrMap.get("value").getTensor)

    val batchMode = sizes.valueAt(1) == -1
    val arraySize = new Array[Int](if (batchMode) sizes.nElement() - 1 else sizes.nElement())
    var i = if (batchMode) 2 else 1
    var k = 0
    while(i <= sizes.nElement()) {
      arraySize(k) = sizes.valueAt(i).toInt
      k += 1
      i += 1
    }
    Reshape[Float](size = arraySize, Some(batchMode))
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object MaxPoolingTF extends TFToBigDL {
  private val graph = {
    (Node("*") -> Node("MaxPool")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val attributes = tfGraph.source.element.getAttrMap

    val (strideH, strideW, ksizeH, ksizeW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(attributes.get("strides").getList.getI(3).toInt == 1, s"not support strides on depth")
      (
        attributes.get("strides").getList.getI(1).toInt,
        attributes.get("strides").getList.getI(2).toInt,
        attributes.get("ksize").getList.getI(1).toInt,
        attributes.get("ksize").getList.getI(2).toInt
      )
    } else if (attributes.get("data_format").getS.toString(Charset.defaultCharset()) == "NCHW") {
      require(attributes.get("strides").getList.getI(2).toInt == 1, s"not support strides on depth")
      (
        attributes.get("strides").getList.getI(2).toInt,
        attributes.get("strides").getList.getI(3).toInt,
        attributes.get("ksize").getList.getI(2).toInt,
        attributes.get("ksize").getList.getI(3).toInt
      )
    } else {
      throw new IllegalArgumentException("no supported data format")
    }

    val (pW, pH) =
      if (attributes.get("padding").getS.toString(Charset.defaultCharset()) == "SAME") {
        require((ksizeW - strideW) % 2 == 0)
        require((ksizeH - strideH) % 2 == 0)
        ((ksizeW - strideW) / 2, (ksizeH - strideH) / 2)
      } else {
        (0, 0)
      }

    val maxpool = SpatialMaxPooling[Float](ksizeW, ksizeH, strideW, strideH, pW, pH)
    maxpool.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object AvgPoolingTF extends TFToBigDL{
  private val graph = {
    (Node("*") -> Node("AvgPool")).graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val attributes = tfGraph.source.element.getAttrMap

    val (strideH, strideW, ksizeH, ksizeW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(attributes.get("strides").getList.getI(3).toInt == 1, s"not support strides on depth")
      (
        attributes.get("strides").getList.getI(1).toInt,
        attributes.get("strides").getList.getI(2).toInt,
        attributes.get("ksize").getList.getI(1).toInt,
        attributes.get("ksize").getList.getI(2).toInt
      )
    } else if (attributes.get("data_format").getS.toString(Charset.defaultCharset()) == "NCHW") {
      require(attributes.get("strides").getList.getI(2).toInt == 1, s"not support strides on depth")
      (
        attributes.get("strides").getList.getI(2).toInt,
        attributes.get("strides").getList.getI(3).toInt,
        attributes.get("ksize").getList.getI(2).toInt,
        attributes.get("ksize").getList.getI(3).toInt
      )
    } else {
      throw new IllegalArgumentException("no supported data format")
    }

    val (pW, pH) =
      if (attributes.get("padding").getS.toString(Charset.defaultCharset()) == "SAME") {
        require((ksizeW - strideW) % 2 == 0)
        require((ksizeH - strideH) % 2 == 0)
        ((ksizeW - strideW) / 2, (ksizeH - strideH) / 2)
      } else {
        (0, 0)
      }

    SpatialAveragePooling[Float](ksizeW, ksizeH, strideW, strideH, pW, pH, countIncludePad = false)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object DropoutTF extends TFToBigDL{
  private val graph = {
    val nodediv = Node("RealDiv")
    val nodeP = Node("Const")
    val nodeadd = Node("Add")
    val noderandom = Node("Add")
    val nodemin = Node("Const")
    val nodesub = Node("Sub")
    val nodemul = Node("Mul")
    val nodedrop = Node("Mul")
    Node("*") -> nodediv -> nodedrop
    nodeP -> nodediv
    nodeP -> nodeadd -> Node("Floor") -> nodedrop
    Node("*") -> Node("Shape") -> Node("RandomUniform") -> nodemul -> noderandom -> nodeadd
    Node("Const") -> nodesub -> nodemul
    nodemin -> nodesub
    nodemin -> noderandom
    nodedrop.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val keepProp = tfGraph.source.prevNodes(0).prevNodes(1).element
      .getAttrMap.get("value").getTensor.getFloatVal(0)

    Dropout[Float](keepProp).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object Placeholder extends TFToBigDL {
  private val graph = Node("Placeholder").graph(reverse = true)

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
      : AbstractModule[Activity, Tensor[Float], Float] = {
    new Input[Float].asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}


object ConstTF extends TFToBigDL {
  private val graph = Node("Const").graph(reverse = true)
  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
   : AbstractModule[Activity, Tensor[Float], Float] = {
    val value = TFToBigDL.toTensor(tfGraph.source.element.getAttrMap.get("value").getTensor)
    Const(value).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ShapeTF extends TFToBigDL {
  private val graph = {
    val node = Node("Shape")
    Node("*") -> node
    node.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : AbstractModule[Activity, Tensor[Float], Float] = {

    new Shape[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object InputTF extends TFToBigDL {
  private val graph = (Node("Const") -> Node("Identity")).graph(reverse = true)

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
    : AbstractModule[Activity, Tensor[Float], Float] = {
    new Input[Float].asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object IdentityTF extends TFToBigDL {
  private val graph = (Node("*") -> Node("Identity")).graph(reverse = true)

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
    : AbstractModule[Activity, Tensor[Float], Float] = {
    new Input[Float].asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object BatchNormTF extends TFToBigDL{
  private val graph = {
    val nodeInput = Node("*")
    val nodeMean1 = Node("Mean")
    val nodeStopGrad = Node("StopGradient")
    val nodeSub1 = Node("Sub")
    val nodeSquare = Node("SquaredDifference")
    val nodeMeanss = Node("Sum")
    val nodeVarss = Node("Sum")
    val nodeShape = Node("Reshape")
    val nodeDivisor = Node("Reciprocal")
    val nodeShiftedMean = Node("Mul")
    val nodeMean2 = Node("Add")
    val nodeMul1 = Node("Mul")
    val nodeVariance = Node("Sub")
    val nodeAdd1 = Node("Add")
    val nodeMul2 = Node("Mul")
    val nodeMul3 = Node("Mul")
    val nodeMul4 = Node("Mul")
    val nodeSub2 = Node("Sub")
    val nodeAdd2 = Node("Add")

    nodeInput -> nodeMul3 -> nodeAdd2
    Node("Const") -> Node("Identity") -> nodeSub2
    nodeInput -> nodeMean1 -> nodeStopGrad -> nodeShape
    Node("Const") -> nodeMean1
    nodeInput -> nodeSub1 -> nodeMeanss -> nodeShiftedMean -> nodeMean2 -> nodeMul4
    nodeStopGrad -> nodeSub1
    nodeInput -> nodeSquare -> nodeVarss -> nodeMul1 -> nodeVariance
    nodeStopGrad -> nodeSquare
    Node("Const") -> nodeDivisor -> nodeShiftedMean -> Node("Square") -> nodeVariance -> nodeAdd1
    Node("Const") -> nodeMeanss -> nodeDivisor -> nodeMul1
    Node("Const") -> nodeVarss -> nodeDivisor
    Node("Const") -> nodeAdd1 -> Node("Rsqrt") -> nodeMul2 -> nodeMul3
    Node("Const") -> Node("Identity") -> nodeMul2 -> nodeMul4 -> nodeSub2 -> nodeAdd2
    Node("Const") -> nodeShape -> nodeMean2
    nodeAdd2.graph(reverse = true)
  }

  override def getInputNodes(tfGraph: DirectedGraph[NodeDef]): Seq[Node[NodeDef]] = {
    Seq(tfGraph.source.prevNodes.head.prevNodes.head)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val nOutput = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
        .prevNodes(1).prevNodes(0).element.getAttrMap.get("value").getTensor.getIntVal(0)

    val weightNode = tfGraph.source.prevNodes(1).prevNodes.head.prevNodes.head.element
    val biasNode = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
      .prevNodes.head.prevNodes.head.element
    val (weights, gradWeights) = getOrSetTensor(weightNode, context)(t => t)
    val (bias, gradBias) = getOrSetTensor(weightNode, context)(t => t)

    val spatialBatchNorm = SpatialBatchNormalization[Float](
      nOutput = nOutput,
      initWeight = weights,
      initBias = bias,
      initGradWeight = weights,
      initGradBias = gradBias
    )
    spatialBatchNorm.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object FillTF extends TFToBigDL{
  private val graph = {
    val nodeFill = Node("Fill")
    Node("*") -> nodeFill
    Node("Const") -> nodeFill
    nodeFill.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val constNode = tfGraph.source.prevNodes(1)
    val const = constNode.element.getAttrMap.get("value").getTensor.getFloatVal(0)

    Fill[Float](const).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object PackTF extends TFToBigDL{
  private val graph = {
    val nodePack = Node("Pack")
    Node("...") -> nodePack
    nodePack.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val dim = tfGraph.source.element.getAttrMap.get("axis").getI.toInt + 1

    Pack[Float](dim).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object UnpackTF extends TFToBigDL{
  private val graph = {
    val nodePack = Node("Unpack")
    Node("*") -> nodePack
    nodePack.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val dim = tfGraph.source.element.getAttrMap.get("axis").getI.toInt + 1
    val index = tfGraph.source.element.getName.split(":").toList match {
      case _::Nil => 1
      case _::i::Nil => i.toInt + 1
    }
    Select[Float](dim, index).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object StrideSliceTF extends TFToBigDL {
  private val graph = {
    val nodeSlice = Node("StridedSlice")
    Node("*") -> nodeSlice
    Node("Const") -> nodeSlice
    Node("Const") -> nodeSlice
    Node("Const") -> nodeSlice
    nodeSlice.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : AbstractModule[Activity, Tensor[Float], Float] = {
    val startNode = tfGraph.source.prevNodes(1)
    val endNode = tfGraph.source.prevNodes(2)
    val strideNode = tfGraph.source.prevNodes(3)

    def getIntArray(node: Node[NodeDef]) = {
      node.element.getAttrMap.get("value").getTensor.getIntValList.asScala.map(_.toInt)
    }

    val start = getIntArray(startNode)
    val end = getIntArray(endNode)
    val stride = getIntArray(strideNode)

    val specs = (start zip end zip stride).zipWithIndex
      .map(elem => (elem._2 + 1, elem._1._1._1 + 1, elem._1._1._2 + 1, elem._1._2)).toArray


    StrideSlice[Float](specs).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}


object ConcatTF extends TFToBigDL{
  private val graph = {
    val nodeConcat = Node("ConcatV2")
    Node("...") -> nodeConcat
    (Node("Const") -> nodeConcat).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val inputNumber = tfGraph.source.element.getAttrMap.get("N").getI.toInt
    val nodeaxis = tfGraph.source.prevNodes(inputNumber)
    val axis = nodeaxis.element.getAttrMap.get("value").getTensor.getIntVal(0)
    val nInputDims = 4

    new JoinTable[Float](dimension = axis + 1, nInputDims = -1)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object AddConstTF1 extends  TFToBigDL{
  private val graph = {
    val nodeAdd = Node("Add")
    Node("Const") -> nodeAdd
    (Node("*") -> nodeAdd).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val value = tfGraph.source.prevNodes.head.element
      .getAttrMap.get("value").getTensor.getFloatVal(0)
    AddConstant[Float](value).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object AddConstTF2 extends  TFToBigDL{
  private val graph = {
    val nodeAdd = Node("Add")
    Node("*") -> nodeAdd
    (Node("Const") -> nodeAdd).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val value = tfGraph.source.prevNodes(1).element
      .getAttrMap.get("value").getTensor.getFloatVal(0)
    AddConstant[Float](value).asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object AddTF extends  TFToBigDL{
  private val graph = {
    val nodeAdd = Node("Add")
    Node("*") -> nodeAdd
    (Node("*") -> nodeAdd).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    CAddTable[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object SoftMaxTF extends  TFToBigDL{
  private val graph = {
    (Node("*") -> Node("Softmax")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    SoftMax[Float]().asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}


object MulTF extends  TFToBigDL{
  private val graph = {
    val nodeMul = Node("Mul")
    Node("Const") -> nodeMul
    (Node("*") -> nodeMul).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val mul = Mul[Float]()
    val scale = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(0).element.getAttrMap.get("value").getTensor)
    require(scale.dim() == 1 && scale.size(1) == 1, s"scale must be one number")
    mul.weight.copy(scale)
    mul.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object ElementWiseMulTF extends  TFToBigDL{
  private val graph = {
    val nodeMul = Node("Mul")
    Node("*") -> nodeMul
    (Node("*") -> nodeMul).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val mul = CMulTable[Float]()
    mul.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object SplitTF extends  TFToBigDL {

  private val graph = {
    val nodeSplit = Node("Split")
    Node("Const") -> nodeSplit
    (Node("*") -> nodeSplit).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : (AbstractModule[Activity, Tensor[Float], Float]) = {
    val numSplit = tfGraph.source.element.getAttrMap.get("num_split").getI.toInt
    val dim = tfGraph.source.prevNodes.head.element
      .getAttrMap.get("value").getTensor.getIntVal(0) + 1
    val index = tfGraph.source.element.getName.split(":").toList match {
      case _::Nil => 1
      case _::i::Nil => i.toInt + 1
    }
    SplitAndSelect[Float](dim, index, numSplit)
      .asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }

}


object PaddingTF extends TFToBigDL{
  private val graph = {
    val nodePad = Node("Pad")
    Node("*") -> nodePad
    (Node("Const") -> nodePad).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : AbstractModule[Activity, Tensor[Float], Float] = {
    val paddings = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(1).element.getAttrMap.get("value").getTensor)
    val pad = ArrayBuffer[Int]()
    val dataFormatMatch = Map("N" -> 0, "H" -> 2, "W" -> 3, "C" -> 1)
    val padding = Sequential[Float]()

    for(i <- 1 to paddings.size(1)) {
      if (paddings(Array(i, 1)) != 0 || paddings(Array(i, 2)) != 0 ) {
        val dim = dataFormatMatch(TFToBigDL.dataFormat.charAt(i-1).toString) + 1
        if (paddings(Array(i, 1)) != 0) {
          padding.add(Padding[Float](dim, -paddings(Array(i, 1)).toInt, 4))
        }
        if (paddings(Array(i, 2)) != 0) {
          padding.add(Padding[Float](dim, paddings(Array(i, 1)).toInt, 4))
        }
      }
    }

    padding.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object MeanTF extends TFToBigDL{
  private val graph = {
    val nodeMean = Node("Mean")
    Node("*") -> nodeMean
    (Node("Const") -> nodeMean).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer(tfGraph: DirectedGraph[NodeDef], context: Context)
  : AbstractModule[Activity, Tensor[Float], Float] = {
    val dims = TFToBigDL.toTensor(
      tfGraph.source.prevNodes(1).element.getAttrMap.get("value").getTensor)
    val dim = ArrayBuffer[Int]()
    val dataFormatMatch = Map("N" -> 0, "H" -> 2, "W" -> 3, "C" -> 1)
    val mean = Sequential[Float]()
    for (i <- 1 to dims.size(1)) {
      dim += dataFormatMatch(TFToBigDL
        .dataFormat.charAt(dims.valueAt(i).toInt).toString) + 1
    }
    dim.foreach(i => mean.add(Mean[Float](i, squeeze = false)))
    mean.asInstanceOf[AbstractModule[Activity, Tensor[Float], Float]]
  }
}

object TFToBigDL {

  /**
   * Get the pattern list.
   * @return
   */
  def patterns : Array[TFToBigDL] = {
    patternList.toArray
  }

  /**
   * Switch endianess to big endian. You should do this when you save the model in a big endian
   * environment. The default endianess is little endian.
   */
  def bigEndian : Unit = endian = ByteOrder.BIG_ENDIAN

  /**
   * Switch endianess to little endian. You should do this when you save the model in a little
   * endian environment. This is the default endianess.
   */
  def littleEndian : Unit = endian = ByteOrder.LITTLE_ENDIAN

  /**
   * Register a new mapping from tensor flow operations to BigDL layer. The mapping is defined as
   * a subclass of TFToBigDL, which defines an operation topology(reversed graph) and how to get
   * constructor parameters from the topology.
   * @param pattern
   */
  def registerPattern(pattern : TFToBigDL): Unit = {
    require(pattern.topology.reverse == true, "the topology should be a reversed graph")
    patternList.append(pattern)
    sortPattern()
  }

  private var endian = ByteOrder.LITTLE_ENDIAN

  var dataFormat : String = "NHWC"

  def dataNCHW : Unit = dataFormat = "NCHW"

  /**
   * Convert a tensorflow tensor proto to BigDL tensor
   * @param tfTensor
   * @return
   */
  private[utils] def toTensor(tfTensor: TensorProto): Tensor[Float] = {
    require(tfTensor.getDtype == DataType.DT_FLOAT || tfTensor.getDtype == DataType.DT_INT32,
      s"Data type ${tfTensor.getDtype} is not supported now")
    val shape = tfTensor.getTensorShape.getDimList.asScala.map(_.getSize.toInt).toArray

    if (shape.product == 1) {
      if (tfTensor.getDtype == DataType.DT_FLOAT) {
        return Tensor[Float](T(tfTensor.getFloatVal(0)))
      } else {
        return Tensor[Float](T(tfTensor.getIntVal(0).toFloat))
      }
    }

    val buffer = ByteBuffer.wrap(tfTensor.getTensorContent.toByteArray)
    buffer.order(endian)

    if (tfTensor.getDtype == DataType.DT_FLOAT) {
      val params = buffer.asFloatBuffer
      val tmp = new Array[Float](params.capacity())
      var j = 0
      while(j < params.capacity()) {
        tmp(j) = params.get(j)
        j += 1
      }
      Tensor(Storage(tmp), 1, shape)
    } else {
      val params = buffer.asIntBuffer
      val tmp = new Array[Float](params.capacity())
      var j = 0
      while(j < params.capacity()) {
        tmp(j) = params.get(j)
        j += 1
      }
      Tensor(Storage(tmp), 1, shape)
    }
  }

  private var patternList : ArrayBuffer[TFToBigDL] = {
    val res = new ArrayBuffer[TFToBigDL]()
    // ElementWiseMulTF must be after MulTF
    res.append(
      FullConnectionTF, DropoutTF, AvgPoolingTF, MaxPoolingTF, ReshapeTF, InputTF,
      TanhTF, ReluTF, SigmoidTF, Conv2D, Placeholder, SqueezeTF, IdentityTF, ConcatTF,
      BatchNormTF, AddConstTF1, AddConstTF2, AddTF, SoftMaxTF, MulTF, ElementWiseMulTF,
      SplitTF, PaddingTF, MeanTF, UnpackTF, StrideSliceTF, ShapeTF, FillTF, PackTF, ConstTF
    )
    res
  }

  sortPattern()

  /**
   * Sort the pattern list to make sure the graph match first should not be a sub-graph of the graph
   * match later
   */
  private def sortPattern() : Unit = {
    // do not calculate size and edges of a graph every time
    val topToNNodes = patternList.map(g => g -> g.topology.size).toMap
    val topToNEdges = patternList.map(g => g -> g.topology.edges).toMap
    patternList = patternList.sortWith((l, r) => {
      if (topToNNodes(l) != topToNNodes(r)) {
        // graph with more nodes comes first
        topToNNodes(l) > topToNNodes(r)
      } else {
        // same node number, graph with more edges come first
        topToNEdges(l) > topToNEdges(r)
      }
    })
  }
}
