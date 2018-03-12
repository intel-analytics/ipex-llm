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

import java.nio.charset.Charset
import java.nio.{ByteBuffer, ByteOrder}
import java.util

import collection.JavaConverters._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}
import com.intel.analytics.bigdl.nn.ops._
import com.intel.analytics.bigdl.tensor._
import org.tensorflow.framework.{AttrValue, DataType, NodeDef, TensorProto}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, DataFormat}
import com.intel.analytics.bigdl.nn.tf._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.FullConnectionTF.getOrSetTensor
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T}
import com.intel.analytics.bigdl.utils.tf.TensorflowToBigDL._

import scala.collection.mutable.ArrayBuffer
import scala.reflect.{ClassTag, classTag}

/**
 * Represent a mapping from tensorflow operations graph to BigDL Module
 */
trait TensorflowToBigDL {

  /**
   * The topology of the tensorflow operation graph
   * @return
   */
  def topology: DirectedGraph[String]

  /**
   * Get the BigDL model
   * @param tfGraph operation graph
   * @param context variables
   * @return (module, input nodes, output nodes)
   */
  def layer[T: ClassTag](
    tfGraph: DirectedGraph[NodeDef],
    context: Context[T],
    byteOrder: ByteOrder
  )(implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T]

  protected def getOrSetTensor[T: ClassTag](
    node: NodeDef, context: Context[T], byteOrder: ByteOrder,
    trans: Option[Seq[(Int, Int)]] = None)(
    implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[T]) = {

    if (context.containsTensor(node.getName)) {
      val result = context(node.getName)
      (result._1, result._2)
    } else {
      var weight = toTensor(node.getAttrMap.get("value").getTensor, byteOrder)
        .asInstanceOf[Tensor[T]]
      trans match {
        case Some(transposes) =>
          for ((first, second) <- transposes) {
            weight = weight.transpose(first, second)
          }
          weight = weight.contiguous()
        case _ =>
      }
      val gradient = Tensor[T](weight.size())
      context.putTensor(node.getName, (weight, gradient, trans))
      (weight, gradient)
    }
  }

  protected def getString(attrMap: util.Map[String, AttrValue], key: String): String = {
    attrMap.get(key).getS.toString(Charset.defaultCharset())
  }

  protected def getInt(attrMap: util.Map[String, AttrValue], key: String): Int = {
    attrMap.get(key).getI.toInt
  }

  protected def getIntList(attrMap: util.Map[String, AttrValue], key: String): Seq[Int] = {
    attrMap.get(key).getList.getIList.asScala.map(_.toInt)
  }

  protected def getBoolean(attrMap: util.Map[String, AttrValue], key: String): Boolean = {
    attrMap.get(key).getB
  }

  protected def getType(attrMap: util.Map[String, AttrValue], key: String): DataType = {
    attrMap.get(key).getType
  }
}

object TensorflowToBigDL {

  /**
   * Represent one input
   */
  val INPUT_PLACEHOLDER: String = "*"

  /**
   * Represent one or many inputs. Note this can only be the first or the last of the input names
   */
  val N_INPUT_PLACEHOLDER: String = "..."

  /**
   * Separate operation name and its output tensor. In tensorflow, if one operation output multiple
   * tensors, the tensor will be referred as Op:n, which n is a integer.
   */
  val TENSOR_SEPARATOR: String = ":"

  /**
   * Get the pattern list.
   * @return
   */
  def patterns: Array[TensorflowToBigDL] = {
    patternList.toArray
  }

  /**
   * Register a new mapping from tensor flow operations to BigDL layer. The mapping is defined as
   * a subclass of TFToBigDL, which defines an operation topology(reversed graph) and how to get
   * constructor parameters from the topology.
   * @param pattern
   */
  def registerPattern(pattern: TensorflowToBigDL): Unit = {
    require(pattern.topology.reverse == true, "the topology should be a reversed graph")
    patternList.append(pattern)
    sortPattern()
  }

  /**
   * Convert a tensorflow tensor proto to BigDL tensor
   * @param tfTensor
   * @return
   */
  private[utils] def toTensor(tfTensor: TensorProto, endian: ByteOrder): Tensor[_] = {
    val shape = tfTensor.getTensorShape.getDimList.asScala.map(_.getSize.toInt).toArray

    /**
     * When there's one element in the tensor. You cannot get the value from byte string
     */
    if (shape.product == 1) {
      if (tfTensor.getDtype == DataType.DT_FLOAT) {
        return Tensor[Float](Storage(Array(tfTensor.getFloatVal(0))), 1, shape)
      }
      if (tfTensor.getDtype == DataType.DT_INT32) {
        return Tensor[Int](Storage(Array(tfTensor.getIntVal(0))), 1, shape)
      }
      if (tfTensor.getDtype == DataType.DT_DOUBLE) {
        return Tensor[Double](Storage(Array(tfTensor.getDoubleVal(0))), 1, shape)
      }
    }

    val buffer = ByteBuffer.wrap(tfTensor.getTensorContent.toByteArray)
    buffer.order(endian)

    if (tfTensor.getDtype == DataType.DT_FLOAT) {
      val params = buffer.asFloatBuffer
      val tmp = new Array[Float](params.capacity())
      var j = 0
      while (j < params.capacity()) {
        tmp(j) = params.get(j)
        j += 1
      }
      return Tensor(Storage(tmp), 1, shape)
    }

    if (tfTensor.getDtype == DataType.DT_INT32) {
      val params = buffer.asIntBuffer
      val tmp = new Array[Int](params.capacity())
      var j = 0
      while (j < params.capacity()) {
        tmp(j) = params.get(j)
        j += 1
      }
      return Tensor(Storage(tmp), 1, shape)
    }

    if (tfTensor.getDtype == DataType.DT_DOUBLE) {
      val params = buffer.asDoubleBuffer()
      val tmp = new Array[Double](params.capacity())
      var j = 0
      while (j < params.capacity()) {
        tmp(j) = params.get(j)
        j += 1
      }
      return Tensor(Storage(tmp), 1, shape)
    }

    throw new UnsupportedOperationException(
      s"Not support load tensorflow tensor when type is ${tfTensor.getDtype}")
  }

  private var patternList : ArrayBuffer[TensorflowToBigDL] = {
    val res = new ArrayBuffer[TensorflowToBigDL]()
    // ElementWiseMulTF must be after MulTF
    res.append(
      FullConnectionTF, DropoutTF, Conv2D, BatchNormTF, Conv1D,
      BatchNormV2NHWCTF, BatchNormV2NCHWTF,
      FullConnectionWithoutBiasTF, Conv2D2,
      Conv2DWithoutBias
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
    val topToNNodes = patternList.map(g => {
      val nodeSize = g.topology.BFS.count(n =>
        n.element != INPUT_PLACEHOLDER && n.element != N_INPUT_PLACEHOLDER)
      g -> nodeSize
    }).toMap

    val topToNEdges = patternList.map(g => {
      val edgeSize = g.topology.BFS.filter(n =>
        n.element != INPUT_PLACEHOLDER && n.element != N_INPUT_PLACEHOLDER)
        .map(_.nextNodes.length).reduce(_ + _)
      g -> edgeSize
    }).toMap

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

object FullConnectionTF extends TensorflowToBigDL{
  private val graph = {
    val add = Node("BiasAdd")
    val mul = Node("MatMul")
    Node("*") -> mul
    Node("Const") -> Node("Identity") -> mul -> add
    Node("Const") -> Node("Identity") -> add
    add.graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph


  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {


    val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.element
    val weightNode = tfGraph.source.prevNodes.head.prevNodes(1).prevNodes.head.element
    val (bias, gradBias) = getOrSetTensor(biasNode, context, byteOrder)
    val (weight, gradWeight) = getOrSetTensor(weightNode, context, byteOrder, Some(Seq((1, 2))))
    Linear[T](inputSize = weight.size(2), outputSize = weight.size(1),
      initWeight = weight, initGradWeight = gradWeight, initBias = bias, initGradBias = gradBias)
      .asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object FullConnectionWithoutBiasTF extends TensorflowToBigDL{
  private val graph = {
    val mul = Node("MatMul")
    Node("*") -> mul
    Node("Const") -> Node("Identity") -> mul
    mul.graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph


  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
     implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {


    val weightNode = tfGraph.source.prevNodes(1).prevNodes.head.element
    val (weight, gradWeight) = getOrSetTensor(weightNode, context, byteOrder, Some(Seq((1, 2))))

    Linear[T](inputSize = weight.size(2), outputSize = weight.size(1), withBias = false,
      initWeight = weight, initGradWeight = gradWeight)
      .asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object Conv1D extends TensorflowToBigDL {
  private val graph = {
    val squeeze = Node("Squeeze")
    val add = Node("BiasAdd")
    val conv = Node("Conv2D")
    val const1 = Node("Const")
    val const2 = Node("Const")
    val expandDimWeight = Node("ExpandDims")
    val expandDimInput = Node("ExpandDims")

    Node("*") -> expandDimInput -> conv
    const1 -> expandDimInput
    Node("Const") -> Node("Identity") -> expandDimWeight -> conv -> squeeze -> add
    const2 -> expandDimWeight
    Node("Const") -> Node("Identity") -> add
    add.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val squeezeNode = tfGraph.source.prevNodes.head
    val convNode = squeezeNode.prevNodes.head

    val attributes = convNode.element.getAttrMap
    val format = getString(attributes, "data_format")
    val strideList = getIntList(attributes, "strides")
    require(strideList.head == 1, s"not support strides on batch")

    val strideW = format match {
      case "NHWC" =>
        strideList(2)
      case "NCHW" =>
        strideList(3)
      case _ =>
        throw new IllegalArgumentException(s"not supported data format: $format")
    }

    val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.element
    val (bias, gradBias) = getOrSetTensor(biasNode, context, byteOrder)

    val weightNode = convNode.prevNodes(1).prevNodes.head.prevNodes.head.element
    val (weights, gradWeights) =
      getOrSetTensor(weightNode, context, byteOrder, Some(Seq((1, 3), (2, 3))))

    val nOuputPlane = weights.size(1)
    val nInputPlane = weights.size(3)
    val kernelW = weights.size(2)

    weights.resize(nOuputPlane, nInputPlane * kernelW)
    gradWeights.resizeAs(weights)

    if (attributes.get("padding").getS.toString(Charset.defaultCharset()) == "SAME") {
      throw new IllegalArgumentException("SAME padding is not supported")
    }

    val tconv = TemporalConvolution[T](
      inputFrameSize = nInputPlane, outputFrameSize = nOuputPlane,
      kernelW = kernelW,
      strideW = strideW,
      initWeight = weights,
      initBias = bias,
      initGradWeight = gradWeights,
      initGradBias = gradBias)

    val result = format match {
      case "NCHW" =>
        val model = Sequential[T]()
        model.add(Transpose(Array((2, 3))))
        model.add(Contiguous())
        model.add(tconv)
        model.add(Transpose(Array((2, 3))))
        model.add(Contiguous())
      case "NHWC" =>
        tconv
    }
    result.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }


}

object Conv2DWithoutBias extends TensorflowToBigDL{
  private val graph = {
    val conv = Node("Conv2D")

    Node("*") -> conv
    Node("Const") -> Node("Identity") -> conv
    conv.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val attributes = tfGraph.source.element.getAttrMap
    val (pW, pH) =
      if (getString(attributes, "padding") == "SAME") {
        (-1, -1)
      } else {
        (0, 0)
      }
    val strideList = getIntList(attributes, "strides")
    require(strideList.head == 1, s"not support strides on batch")

    val format = getString(attributes, "data_format")
    val conv = format match {
      case "NHWC" =>
        require(strideList(3) == 1, s"not support strides on depth")
        val strideW = strideList(1)
        val strideH = strideList(2)
        val weightNode = tfGraph.source.prevNodes(1).prevNodes.head.element
        val (weights, gradWeights) = getOrSetTensor(weightNode, context, byteOrder)
        val nOuputPlane = weights.size(4)
        val nInputPlane = weights.size(3)
        val kernelH = weights.size(1)
        val kernelW = weights.size(2)
        SpatialConvolution[T](
          nInputPlane = nInputPlane, nOutputPlane = nOuputPlane,
          kernelW = kernelW, kernelH = kernelH,
          strideW = strideW, strideH = strideH,
          padW = pW, padH = pH,
          initWeight = weights, initGradWeight = gradWeights,
          format = DataFormat.NHWC,
          withBias = false
        )

      case "NCHW" =>
        require(strideList(1) == 1, s"not support strides on depth")
        val strideW = strideList(2)
        val strideH = strideList(3)
        val weightNode = tfGraph.source.prevNodes(1).prevNodes.head.element
        val (weights, gradWeights) =
          getOrSetTensor(weightNode, context, byteOrder, Some(Seq((1, 4), (2, 3), (3, 4))))
        val nOuputPlane = weights.size(1)
        val nInputPlane = weights.size(2)
        val kernelH = weights.size(3)
        val kernelW = weights.size(4)
        SpatialConvolution[T](
          nInputPlane = nInputPlane, nOutputPlane = nOuputPlane,
          kernelW = kernelW, kernelH = kernelH,
          strideW = strideW, strideH = strideH,
          padW = pW, padH = pH,
          initWeight = weights, initGradWeight = gradWeights,
          format = DataFormat.NCHW,
          withBias = false
        )
      case _ =>
        throw new IllegalArgumentException(s"not supported data format: $format")
    }
    conv.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object Conv2D extends TensorflowToBigDL{
  private val graph = {
    val add = Node("BiasAdd")
    val conv = Node("Conv2D")

    Node("*") -> conv
    Node("Const") -> Node("Identity") -> conv -> add
    Node("Const") -> Node("Identity") -> add
    add.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
    context: Context[T],
    byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val attributes = tfGraph.source.prevNodes.head.element.getAttrMap
    val (pW, pH) =
      if (getString(attributes, "padding") == "SAME") {
        (-1, -1)
      } else {
        (0, 0)
      }
    val strideList = getIntList(attributes, "strides")
    require(strideList.head == 1, s"not support strides on batch")

    val format = getString(attributes, "data_format")
    val conv = format match {
      case "NHWC" =>
        require(strideList(3) == 1, s"not support strides on depth")
        val strideW = strideList(1)
        val strideH = strideList(2)
        val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.element
        val (bias, gradBias) = getOrSetTensor(biasNode, context, byteOrder)
        val weightNode = tfGraph.source.prevNodes.head.prevNodes(1).prevNodes.head.element
        val (weights, gradWeights) = getOrSetTensor(weightNode, context, byteOrder)
        val nOuputPlane = weights.size(4)
        val nInputPlane = weights.size(3)
        val kernelH = weights.size(1)
        val kernelW = weights.size(2)
        SpatialConvolution[T](
          nInputPlane = nInputPlane, nOutputPlane = nOuputPlane,
          kernelW = kernelW, kernelH = kernelH,
          strideW = strideW, strideH = strideH,
          padW = pW, padH = pH,
          initWeight = weights,
          initBias = bias,
          initGradWeight = gradWeights,
          initGradBias = gradBias, format = DataFormat.NHWC)

      case "NCHW" =>
        require(strideList(1) == 1, s"not support strides on depth")
        val strideW = strideList(2)
        val strideH = strideList(3)
        val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.element
        val (bias, gradBias) = getOrSetTensor(biasNode, context, byteOrder)

        val weightNode = tfGraph.source.prevNodes.head.prevNodes(1).prevNodes.head.element
        val (weights, gradWeights) =
          getOrSetTensor(weightNode, context, byteOrder, Some(Seq((1, 4), (2, 3), (3, 4))))
        val nOuputPlane = weights.size(1)
        val nInputPlane = weights.size(2)
        val kernelH = weights.size(3)
        val kernelW = weights.size(4)
        SpatialConvolution[T](
          nInputPlane = nInputPlane, nOutputPlane = nOuputPlane,
          kernelW = kernelW, kernelH = kernelH,
          strideW = strideW, strideH = strideH,
          padW = pW, padH = pH,
          initWeight = weights,
          initBias = bias,
          initGradWeight = gradWeights,
          initGradBias = gradBias, format = DataFormat.NCHW)
      case _ =>
        throw new IllegalArgumentException(s"not supported data format: $format")
    }
    conv.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object Conv2D2 extends TensorflowToBigDL{
  private val graph = {
    val add = Node("Add")
    val conv = Node("Conv2D")
    val reshape = Node("Reshape")

    Node("*") -> conv
    Node("Const") -> Node("Identity") -> conv -> add
    Node("Const") -> Node("Identity") -> reshape
    Node("Const") -> reshape
    reshape -> add

    add.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
    context: Context[T],
    byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val attributes = tfGraph.source.prevNodes(0).element.getAttrMap
    val strideList = getIntList(attributes, "strides")
    val format = getString(attributes, "data_format")
    require(strideList.head == 1, s"not support strides on batch")
    require(format == "NCHW", "NCHW should be used for this sub-graph")

    require(strideList(1) == 1, s"not support strides on depth")
    val (strideH, strideW) = (strideList(2), strideList(3))

    val biasNode = tfGraph.source.prevNodes(1).prevNodes(0).prevNodes.head.element
    val (bias, gradBias) = getOrSetTensor(biasNode, context, byteOrder)

    val weightNode = tfGraph.source.prevNodes.head.prevNodes(1).prevNodes.head.element
    val (weights, gradWeights) =
      getOrSetTensor(weightNode, context, byteOrder, Some(Seq((1, 4), (2, 3), (3, 4))))

    val nOuputPlane = weights.size(1)
    val nInputPlane = weights.size(2)
    val kernelH = weights.size(3)
    val kernelW = weights.size(4)

    val (pW, pH) =
      if (getString(attributes, "padding") == "SAME") {
        (-1, -1)
      } else {
        (0, 0)
      }

    SpatialConvolution[T](
      nInputPlane = nInputPlane, nOutputPlane = nOuputPlane,
      kernelW = kernelW, kernelH = kernelH,
      strideW = strideW, strideH = strideH,
      padW = pW, padH = pH,
      initWeight = weights,
      initBias = bias,
      initGradWeight = gradWeights,
      initGradBias = gradBias).asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object DropoutTF extends TensorflowToBigDL{
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

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val keepProp = tfGraph.source.prevNodes(0).prevNodes(1).element
      .getAttrMap.get("value").getTensor.getFloatVal(0)
    val model = Sequential[T]()
    model.add(SelectTable(1))
    model.add(Dropout[T](keepProp))
    model.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object BatchNormV2NCHWTF extends TensorflowToBigDL{
  private val graph = {
    val nodeInput = Node("*")
    val nodeMean = Node("Mean")
    val nodeStopGrad = Node("StopGradient")
    val nodeSub = Node("Sub")
    val nodeSquare = Node("SquaredDifference")
    val nodeShiftedMean = Node("Mean")
    val node_mean = Node("Add")
    val nodeMean_1 = Node("Mean")
    val nodeVariance = Node("Sub")
    val nodeAdd = Node("Add")
    val nodeMul = Node("Mul")
    val nodeMul_1 = Node("Mul")
    val nodeMul_2 = Node("Mul")
    val node_sub = Node("Sub")
    val nodeAdd_1 = Node("Add")
    val nodeSqueeze_1 = Node("Squeeze")
    val nodeSqueeze = Node("Squeeze")
    val reshape1 = Node("Reshape")
    val reshape = Node("Reshape")
    val reshape2 = Node("Reshape")
    val reshape3 = Node("Reshape")

    nodeInput -> nodeMul_1 -> nodeAdd_1
    Node("Const") -> Node("Identity") -> reshape2 -> node_sub
    nodeInput -> nodeSub -> nodeShiftedMean -> node_mean -> nodeSqueeze -> reshape -> nodeMul_2
    nodeInput -> nodeMean -> nodeStopGrad -> node_mean
    Node("Const") -> nodeMean
    nodeStopGrad -> nodeSub
    nodeInput -> nodeSquare -> nodeMean_1 -> nodeVariance
    Node("Const") -> nodeMean_1
    nodeStopGrad -> nodeSquare
    Node("Const") -> nodeShiftedMean -> Node("Square") ->
      nodeVariance -> nodeSqueeze_1 -> reshape1 -> nodeAdd
    Node("Const") -> nodeAdd -> Node("Rsqrt") -> nodeMul -> nodeMul_1
    Node("Const") -> Node("Identity") -> reshape3 -> nodeMul -> nodeMul_2 -> node_sub -> nodeAdd_1
    Node("Const") -> reshape
    Node("Const") -> reshape1
    Node("Const") -> reshape2
    Node("Const") -> reshape3
    nodeAdd_1.graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
          implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.prevNodes.head.prevNodes.head.element
    val weightNode = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
      .prevNodes(1).prevNodes.head.prevNodes.head.element
    val (weights, gradWeights) = getOrSetTensor[T](weightNode, context, byteOrder)
    val (bias, gradBias) = getOrSetTensor[T](biasNode, context, byteOrder)

    val batchNorm = SpatialBatchNormalization[T](
      nOutput = weights.size(1),
      initWeight = weights,
      initBias = bias,
      initGradWeight = gradWeights,
      initGradBias = gradBias
    )

    val model = Sequential[T]()
    model.add(SelectTable(1))
    model.add(batchNorm)
    model.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object BatchNormV2NHWCTF extends TensorflowToBigDL{
  private val graph = {
    val nodeInput = Node("*")
    val nodeMean = Node("Mean")
    val nodeStopGrad = Node("StopGradient")
    val nodeSub = Node("Sub")
    val nodeSquare = Node("SquaredDifference")
    val nodeShiftedMean = Node("Mean")
    val node_mean = Node("Add")
    val nodeMean_1 = Node("Mean")
    val nodeVariance = Node("Sub")
    val nodeAdd = Node("Add")
    val nodeMul = Node("Mul")
    val nodeMul_1 = Node("Mul")
    val nodeMul_2 = Node("Mul")
    val node_sub = Node("Sub")
    val nodeAdd_1 = Node("Add")
    val nodeSqueeze_1 = Node("Squeeze")
    val nodeSqueeze = Node("Squeeze")

    nodeInput -> nodeMul_1 -> nodeAdd_1
    Node("Const") -> Node("Identity") -> node_sub
    nodeInput -> nodeSub -> nodeShiftedMean -> node_mean -> nodeSqueeze -> nodeMul_2
    nodeInput -> nodeMean -> nodeStopGrad -> node_mean
    Node("Const") -> nodeMean
    nodeStopGrad -> nodeSub
    nodeInput -> nodeSquare -> nodeMean_1 -> nodeVariance
    Node("Const") -> nodeMean_1
    nodeStopGrad -> nodeSquare
    Node("Const") -> nodeShiftedMean -> Node("Square") -> nodeVariance -> nodeSqueeze_1 -> nodeAdd
    Node("Const") -> nodeAdd -> Node("Rsqrt") -> nodeMul -> nodeMul_1
    Node("Const") -> Node("Identity") -> nodeMul -> nodeMul_2 -> node_sub -> nodeAdd_1
    nodeAdd_1.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
               implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.prevNodes.head.element
    val weightNode = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
      .prevNodes(1).prevNodes.head.element
    val (weights, gradWeights) = getOrSetTensor[T](weightNode, context, byteOrder)
    val (bias, gradBias) = getOrSetTensor[T](biasNode, context, byteOrder)

    val batchNorm = SpatialBatchNormalization[T](
      nOutput = weights.size(1),
      initWeight = weights,
      initBias = bias,
      initGradWeight = gradWeights,
      initGradBias = gradBias
    )

    val layer = Sequential[T]()
    layer.add(SelectTable(1))
    layer.add(Transpose(Array((2, 4))))
    layer.add(Contiguous())
    layer.add(batchNorm)
    layer.add(Transpose(Array((2, 4))))
    layer.add(Contiguous())

    layer.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

object BatchNormTF extends TensorflowToBigDL{
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

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {

    val nOutput = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
        .prevNodes(1).prevNodes(0).element.getAttrMap.get("value").getTensor.getIntVal(0)

    val weightNode = tfGraph.source.prevNodes(1).prevNodes.head.prevNodes.head.element
    val biasNode = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
      .prevNodes.head.prevNodes.head.element
    val (weights, gradWeights) = getOrSetTensor[T](weightNode, context, byteOrder)
    val (bias, gradBias) = getOrSetTensor[T](weightNode, context, byteOrder)

    val batchNorm = SpatialBatchNormalization[T](
      nOutput = nOutput,
      initWeight = weights,
      initBias = bias,
      initGradWeight = gradWeights,
      initGradBias = gradBias
    )
    val model = Sequential[T]()
    model.add(SelectTable(1))
    model.add(batchNorm)
    model.asInstanceOf[AbstractModule[Activity, Activity, T]]
  }
}

