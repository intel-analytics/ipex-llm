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

import collection.JavaConverters._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.tensorflow.framework.{DataType, NodeDef, TensorProto}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.tf._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{DirectedGraph, Node, T}
import com.intel.analytics.bigdl.utils.tf.TensorflowLoader.Context
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
  )(implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T]

  protected def getOrSetTensor[T: ClassTag](
    node: NodeDef, context: Context[T], byteOrder: ByteOrder)(f: Tensor[T] => Tensor[T])(
    implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[T]) = {

    if (context.contains(node)) {
      context(node)
    } else {
      val weight = f(toTensor[T](node.getAttrMap.get("value").getTensor, byteOrder)).contiguous()
      val gradient = Tensor[T](weight.size())
      context.put(node, (weight, gradient))
      (weight, gradient)
    }
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

  var dataFormat: String = "NHWC"

  def dataNCHW: Unit = dataFormat = "NCHW"

  /**
   * Convert a tensorflow tensor proto to BigDL tensor
   * @param tfTensor
   * @return
   */
  private[utils] def toTensor[T: ClassTag](tfTensor: TensorProto, endian: ByteOrder)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {

    require(
      tfTensor.getDtype == DataType.DT_FLOAT ||
        tfTensor.getDtype == DataType.DT_DOUBLE ||
        tfTensor.getDtype == DataType.DT_INT32,
      s"Data type ${tfTensor.getDtype} is not supported now")

    val shape = tfTensor.getTensorShape.getDimList.asScala.map(_.getSize.toInt).toArray

    /**
     * When there's one element in the tensor. You cannot get the value from byte string
     */
    if (shape.product == 1) {
      if (classTag[T] == classTag[Float]) {
        if (tfTensor.getDtype == DataType.DT_FLOAT) {
          return Tensor[Float](T(tfTensor.getFloatVal(0))).asInstanceOf[Tensor[T]]
        }

        if (tfTensor.getDtype == DataType.DT_INT32) {
          return Tensor[Float](T(tfTensor.getIntVal(0).toFloat)).asInstanceOf[Tensor[T]]
        }

        throw new IllegalArgumentException("Can not convert double to float")
      } else if (classTag[T] == classTag[Double]) {
        if (tfTensor.getDtype == DataType.DT_DOUBLE) {
          return Tensor[Float](T(tfTensor.getDoubleVal(0))).asInstanceOf[Tensor[T]]
        }

        if (tfTensor.getDtype == DataType.DT_FLOAT) {
          return Tensor[Float](T(tfTensor.getFloatVal(0).toDouble)).asInstanceOf[Tensor[T]]
        }

        if (tfTensor.getDtype == DataType.DT_INT32) {
          return Tensor[Float](T(tfTensor.getIntVal(0).toDouble)).asInstanceOf[Tensor[T]]
        }
      }
    }

    val buffer = ByteBuffer.wrap(tfTensor.getTensorContent.toByteArray)
    buffer.order(endian)

    if (classTag[T] == classTag[Float]) {
      if (tfTensor.getDtype == DataType.DT_FLOAT) {
        val params = buffer.asFloatBuffer
        val tmp = new Array[Float](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor(Storage(tmp), 1, shape).asInstanceOf[Tensor[T]]
      } else if (tfTensor.getDtype == DataType.DT_INT32) {
        val params = buffer.asIntBuffer
        val tmp = new Array[Float](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor(Storage(tmp), 1, shape).asInstanceOf[Tensor[T]]
      } else {
        throw new IllegalArgumentException("Can not convert double to float")
      }
    } else if (classTag[T] == classTag[Double]) {
      if (tfTensor.getDtype == DataType.DT_FLOAT) {
        val params = buffer.asFloatBuffer
        val tmp = new Array[Double](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor(Storage(tmp), 1, shape).asInstanceOf[Tensor[T]]
      } else if (tfTensor.getDtype == DataType.DT_INT32) {
        val params = buffer.asIntBuffer
        val tmp = new Array[Double](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor(Storage(tmp), 1, shape).asInstanceOf[Tensor[T]]
      } else if (tfTensor.getDtype == DataType.DT_DOUBLE) {
        val params = buffer.asDoubleBuffer()
        val tmp = new Array[Double](params.capacity())
        var j = 0
        while (j < params.capacity()) {
          tmp(j) = params.get(j)
          j += 1
        }
        Tensor(Storage(tmp), 1, shape).asInstanceOf[Tensor[T]]
      } else {
        throw new IllegalArgumentException(s"Data type ${tfTensor.getDtype} is not supported now")
      }
    } else {
      throw new IllegalArgumentException("Only support Float/Double")
    }
  }

  private var patternList : ArrayBuffer[TensorflowToBigDL] = {
    val res = new ArrayBuffer[TensorflowToBigDL]()
    // ElementWiseMulTF must be after MulTF
    res.append(
      FullConnectionTF, DropoutTF, AvgPoolingTF, MaxPoolingTF, ReshapeTF, InputTF,
      TanhTF, ReluTF, SigmoidTF, Conv2D, Placeholder, SqueezeTF, IdentityTF, ConcatTF,
      BatchNormTF, AddConstTF1, AddConstTF2, AddTF, SoftMaxTF, ElementWiseMulTF, MulTF,
      SplitTF, PaddingTF, MeanTF, UnpackTF, StrideSliceTF, ShapeTF, FillTF, PackTF, ConstTF,
      Flatten
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

  /**
   * This method is just for test purpose. Do not use the bigdl.saveNHWC for real use case
   * @return
   */
  private[tf] def processDims(dim: Int): Int = {
    if (System.getProperty("bigdl.enableNHWC", "false").toBoolean) {
      // exchange the dims as BigDL only support NCHW now
      if (dim == 1) return 2
      if (dim == 2) return 3
      if (dim == 3) return 1
      dim
    } else {
      dim
    }
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
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {


    val biasNode = tfGraph.source.prevNodes(1).prevNodes.head.element
    val weightNode = tfGraph.source.prevNodes.head.prevNodes(1).prevNodes.head.element
    val (bias, gradBias) = getOrSetTensor(biasNode, context, byteOrder)(t => t)
    val (weight, gradWeight) = getOrSetTensor(weightNode, context, byteOrder) { t =>
      t.transpose(1, 2)
    }

    Linear[T](inputSize = weight.size(2), outputSize = weight.size(1),
      initWeight = weight, initGradWeight = gradWeight, initBias = bias, initGradBias = gradBias)
      .asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object  SqueezeTF extends TensorflowToBigDL {
  private val graph = (Node("*") -> Node("Squeeze")).graph(reverse = true)
  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val dims = tfGraph.source.element.getAttrOrThrow("squeeze_dims").getList().getIList()
      .asScala.map(_.toInt).toArray.map(processDims(_))

    Squeeze[T](dims, batchMode = true).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
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
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {


    val attributes = tfGraph.source.prevNodes(0).element.getAttrMap
    require(attributes.get("strides").getList.getI(0).toInt == 1, s"not support strides on batch")

    val (strideH, strideW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(System.getProperty("bigdl.enableNHWC", "false").toBoolean, "Not support NHWC")
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
    val (bias, gradBias) = getOrSetTensor(biasNode, context, byteOrder)(t => t)

    val weightNode = tfGraph.source.prevNodes.head.prevNodes(1).prevNodes.head.element
    val (weights, gradWeights) = getOrSetTensor(weightNode, context, byteOrder) { t =>
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

    SpatialConvolution[T](
      nInputPlane = nInputPlane, nOutputPlane = nOuputPlane,
      kernelW = kernelW, kernelH = kernelH,
      strideW = strideW, strideH = strideH,
      padW = pW, padH = pH,
      initWeight = weights,
      initBias = bias,
      initGradWeight = gradWeights,
      initGradBias = gradBias).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object ReluTF extends  TensorflowToBigDL {
  private val graph = {
    (Node("*") -> Node("Relu")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    ReLU[T]().asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object TanhTF extends  TensorflowToBigDL{
  private val graph = {
    (Node("*") -> Node("Tanh")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {


    Tanh[T]().asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object SigmoidTF extends  TensorflowToBigDL{
  private val graph = {
    (Node("*") -> Node("Sigmoid")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    Sigmoid[T]().asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object ReshapeTF extends TensorflowToBigDL {
  private val graph = {
    val nodeReshape = Node("Reshape")
    Node("*") -> nodeReshape
    Node("Const") -> nodeReshape
    nodeReshape.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val sizes = TensorflowToBigDL.toTensor(
      tfGraph.source.prevNodes(1).element.getAttrMap.get("value").getTensor, byteOrder)

    val batchMode = sizes.valueAt(1) == -1
    val arraySize = new Array[Int](if (batchMode) sizes.nElement() - 1 else sizes.nElement())
    var i = if (batchMode) 2 else 1
    var k = 0
    while(i <= sizes.nElement()) {
      arraySize(k) = ev.toType[Int](sizes.valueAt(i))
      k += 1
      i += 1
    }
    Reshape[T](size = arraySize, Some(batchMode))
      .asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object MaxPoolingTF extends TensorflowToBigDL {
  private val graph = {
    (Node("*") -> Node("MaxPool")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val attributes = tfGraph.source.element.getAttrMap

    val (strideH, strideW, ksizeH, ksizeW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(System.getProperty("bigdl.enableNHWC", "false").toBoolean, "Not support NHWC")
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

    SpatialMaxPooling[T](ksizeW, ksizeH, strideW, strideH, pW, pH)
      .asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object AvgPoolingTF extends TensorflowToBigDL{
  private val graph = {
    (Node("*") -> Node("AvgPool")).graph(reverse = true)
  }
  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val attributes = tfGraph.source.element.getAttrMap

    val (strideH, strideW, ksizeH, ksizeW) = if (attributes.get("data_format").getS
      .toString(Charset.defaultCharset()) == "NHWC") {
      require(System.getProperty("bigdl.enableNHWC", "false").toBoolean, "Not support NHWC")
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

    SpatialAveragePooling[T](ksizeW, ksizeH, strideW, strideH, pW, pH, countIncludePad = false)
      .asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
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
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val keepProp = tfGraph.source.prevNodes(0).prevNodes(1).element
      .getAttrMap.get("value").getTensor.getFloatVal(0)

    Dropout[T](keepProp).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object Placeholder extends TensorflowToBigDL {
  private val graph = Node("Placeholder").graph(reverse = true)

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {
    Input[T].element.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}


object ConstTF extends TensorflowToBigDL {
  private val graph = Node("Const").graph(reverse = true)
  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val value = TensorflowToBigDL
      .toTensor(tfGraph.source.element.getAttrMap.get("value").getTensor, byteOrder)
    Const(value).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object ShapeTF extends TensorflowToBigDL {
  private val graph = {
    val node = Node("Shape")
    Node("*") -> node
    node.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {


    Shape[T]().asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object InputTF extends TensorflowToBigDL {
  private val graph = (Node("Const") -> Node("Identity")).graph(reverse = true)

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    Input[T].element.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object IdentityTF extends TensorflowToBigDL {
  private val graph = (Node("*") -> Node("Identity")).graph(reverse = true)

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    Input[T].element.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
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
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val nOutput = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
        .prevNodes(1).prevNodes(0).element.getAttrMap.get("value").getTensor.getIntVal(0)

    val weightNode = tfGraph.source.prevNodes(1).prevNodes.head.prevNodes.head.element
    val biasNode = tfGraph.source.prevNodes(1).prevNodes(1).prevNodes(1)
      .prevNodes.head.prevNodes.head.element
    val (weights, gradWeights) = getOrSetTensor[T](weightNode, context, byteOrder)(t => t)
    val (bias, gradBias) = getOrSetTensor[T](weightNode, context, byteOrder)(t => t)

    SpatialBatchNormalization[T](
      nOutput = nOutput,
      initWeight = weights,
      initBias = bias,
      initGradWeight = gradWeights,
      initGradBias = gradBias
    ).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object FillTF extends TensorflowToBigDL{
  private val graph = {
    val nodeFill = Node("Fill")
    Node("*") -> nodeFill
    Node("Const") -> nodeFill
    nodeFill.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val constNode = tfGraph.source.prevNodes(1)
    val const = constNode.element.getAttrMap.get("value").getTensor.getFloatVal(0)

    Fill[T](const).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object PackTF extends TensorflowToBigDL{
  private val graph = {
    val nodePack = Node("Pack")
    Node("...") -> nodePack
    nodePack.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {
    val dim = processDims(tfGraph.source.element.getAttrMap.get("axis").getI.toInt + 1)

    Pack[T](dim).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object UnpackTF extends TensorflowToBigDL{
  private val graph = {
    val nodePack = Node("Unpack")
    Node("*") -> nodePack
    nodePack.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val dim = processDims(tfGraph.source.element.getAttrMap.get("axis").getI.toInt + 1)
    val index = tfGraph.source.element.getName.split(":").toList match {
      case _::Nil => 1
      case _::i::Nil => i.toInt + 1
    }
    Select[T](dim, index).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object StrideSliceTF extends TensorflowToBigDL {
  private val graph = {
    val nodeSlice = Node("StridedSlice")
    Node("*") -> nodeSlice
    Node("Const") -> nodeSlice
    Node("Const") -> nodeSlice
    Node("Const") -> nodeSlice
    nodeSlice.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

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


    StrideSlice[T](specs).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}


object ConcatTF extends TensorflowToBigDL{
  private val graph = {
    val nodeConcat = Node("ConcatV2")
    Node("...") -> nodeConcat
    (Node("Const") -> nodeConcat).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val inputNumber = tfGraph.source.element.getAttrMap.get("N").getI.toInt
    val nodeaxis = tfGraph.source.prevNodes(inputNumber)
    val axis = processDims(
      nodeaxis.element.getAttrMap.get("value").getTensor.getIntVal(0))
    val nInputDims = 4

    JoinTable[T](dimension = axis + 1, nInputDims = -1)
      .asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object Flatten extends TensorflowToBigDL {
  private val graph = {
    val reshapeNode = Node("Reshape")
    val concatNode = Node("ConcatV2")
    val sliceNode = Node("Slice")
    val expandNode = Node("ExpandDims")
    val prodNode = Node("Prod")
    val sliceNode1 = Node("Slice")
    val shapeNode = Node("Const")
    val beginNode = Node("Const")
    val sizeNode = Node("Const")
    val beginNode1 = Node("Const")
    val sizeNode1 = Node("Const")
    val constNode = Node("Const")
    val dimNode = Node("Const")
    val axisNode = Node("Const")

    shapeNode -> sliceNode
    beginNode -> sliceNode
    sizeNode -> sliceNode

    shapeNode -> sliceNode1
    beginNode1 -> sliceNode1
    sizeNode1 -> sliceNode1

    sliceNode1 -> prodNode
    constNode -> prodNode

    prodNode -> expandNode
    dimNode -> expandNode

    sliceNode -> concatNode
    expandNode -> concatNode
    axisNode -> concatNode

    Node("*") -> reshapeNode
    concatNode -> reshapeNode
    reshapeNode.graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
      context: Context[T],
      byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {
    val shapetfTensor = tfGraph.source.prevNodes(1).prevNodes(0).prevNodes(0).element
      .getAttrMap.get("value").getTensor
    val sizes = TensorflowToBigDL.toTensor(shapetfTensor, byteOrder)
    val batchMode = false

    val arraySize = Array(
      ev.toType[Int](sizes.valueAt(1)),
      {
        var prod = 1
        var i = 2
        while(i <= sizes.nElement()) {
          prod = prod * ev.toType[Int](sizes.valueAt(i))
          i = i + 1
        }
        prod
      }
    )

    Reshape[T](size = arraySize, Some(batchMode))
      .asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object AddConstTF1 extends  TensorflowToBigDL{
  private val graph = {
    val nodeAdd = Node("Add")
    Node("Const") -> nodeAdd
    (Node("*") -> nodeAdd).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {
    val value = tfGraph.source.prevNodes.head.element
      .getAttrMap.get("value").getTensor.getFloatVal(0)
    AddConstant[T](value).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object AddConstTF2 extends  TensorflowToBigDL{
  private val graph = {
    val nodeAdd = Node("Add")
    Node("*") -> nodeAdd
    (Node("Const") -> nodeAdd).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val value = tfGraph.source.prevNodes(1).element
      .getAttrMap.get("value").getTensor.getFloatVal(0)
    AddConstant[T](value).asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object AddTF extends  TensorflowToBigDL{
  private val graph = {
    val nodeAdd = Node("Add")
    Node("*") -> nodeAdd
    (Node("*") -> nodeAdd).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    CAddTable[T]().asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object SoftMaxTF extends  TensorflowToBigDL{
  private val graph = {
    (Node("*") -> Node("Softmax")).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph
  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    SoftMax[T]().asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}


object MulTF extends  TensorflowToBigDL{
  private val graph = {
    val nodeMul = Node("Mul")
    Node("Const") -> nodeMul
    (Node("*") -> nodeMul).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val scale = TensorflowToBigDL.toTensor(
      tfGraph.source.prevNodes(0).element.getAttrMap.get("value").getTensor, byteOrder)
    require(scale.dim() == 1 && scale.size(1) == 1, s"scale must be one number")
    val mul = MulConstant[T](ev.toType[Double](scale.valueAt(1)))
    mul.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object ElementWiseMulTF extends  TensorflowToBigDL{
  private val graph = {
    val nodeMul = Node("Mul")
    Node("*") -> nodeMul
    (Node("*") -> nodeMul).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    CMulTable[T]().asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object SplitTF extends  TensorflowToBigDL {

  private val graph = {
    val nodeSplit = Node("Split")
    Node("Const") -> nodeSplit
    (Node("*") -> nodeSplit).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val numSplit = tfGraph.source.element.getAttrMap.get("num_split").getI.toInt
    val dim = tfGraph.source.prevNodes.head.element
      .getAttrMap.get("value").getTensor.getIntVal(0) + 1
    val index = tfGraph.source.element.getName.split(":").toList match {
      case _::Nil => 1
      case _::i::Nil => i.toInt + 1
    }
    SplitAndSelect[T](dim, index, numSplit)
      .asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }

}


object PaddingTF extends TensorflowToBigDL{
  private val graph = {
    val nodePad = Node("Pad")
    Node("*") -> nodePad
    (Node("Const") -> nodePad).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val paddings = TensorflowToBigDL.toTensor(
      tfGraph.source.prevNodes(1).element.getAttrMap.get("value").getTensor, byteOrder)
    val pad = ArrayBuffer[Int]()
    val padding = Sequential[T]()

    for(i <- 1 to paddings.size(1)) {
      if (paddings.valueAt(i, 1) != 0 || paddings.valueAt(i, 2) != 0 ) {
        val dim = processDims(i - 1) + 1
        if (paddings(Array(i, 1)) != 0) {
          padding.add(Padding[T](dim, -ev.toType[Int](paddings.valueAt(i, 1)), 4))
        }
        if (paddings(Array(i, 2)) != 0) {
          padding.add(Padding[T](dim, ev.toType[Int](paddings.valueAt(i, 2)), 4))
        }
      }
    }

    padding.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}

object MeanTF extends TensorflowToBigDL{
  private val graph = {
    val nodeMean = Node("Mean")
    Node("*") -> nodeMean
    (Node("Const") -> nodeMean).graph(reverse = true)
  }

  override def topology: DirectedGraph[String] = graph

  override def layer[T: ClassTag](tfGraph: DirectedGraph[NodeDef],
                                  context: Context[T],
                                  byteOrder: ByteOrder)(
    implicit ev: TensorNumeric[T]): AbstractModule[Activity, Tensor[T], T] = {

    val dims = TensorflowToBigDL.toTensor(
      tfGraph.source.prevNodes(1).element.getAttrMap.get("value").getTensor, byteOrder)
    val dim = ArrayBuffer[Int]()
    val mean = Sequential[T]()
    for (i <- 1 to dims.size(1)) {
      dim += processDims(ev.toType[Int](dims.valueAt(i))) + 1
    }
    dim.foreach(i => mean.add(Mean[T](i, squeeze = false)))
    mean.asInstanceOf[AbstractModule[Activity, Tensor[T], T]]
  }
}
