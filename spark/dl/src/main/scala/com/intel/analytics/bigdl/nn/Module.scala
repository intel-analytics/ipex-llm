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
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.File
import com.intel.analytics.bigdl.utils.caffe.CaffeLoader
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.bigdl.utils.tf.{TensorflowDataFormat, TensorflowLoader}

import scala.reflect.ClassTag

object Module {
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
   * @tparam T numeric type
   * @return model loaded from path
   */
  def loadModule[T: ClassTag](path : String)(implicit ev: TensorNumeric[T])
  : AbstractModule[Activity, Activity, T] = {
    ModuleLoader.loadFromFile(path)
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
   * @param file where is the protobuf model file
   * @param inputs input node names
   * @param outputs output node names, the output tensor order is same with the node order
   * @param byteOrder byte order in the tensorflow file. The default value is little endian
   * @return BigDL model
   */
  def loadTF[T: ClassTag](file: String, inputs: Seq[String], outputs: Seq[String],
            byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN)(
    implicit ev: TensorNumeric[T]): Module[T] = {

    TensorflowLoader.load(file, inputs, outputs, byteOrder)
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

  def isCompact[@specialized(Float, Double) T: ClassTag](paramters: Array[Tensor[T]])(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    require(paramters.length > 0,
      "The length of paramters should >= 0" +
      "parameter length" +
        s" ${paramters.length}")
    var i = 1
    val storage = paramters(0).storage()
    var length = paramters(0).nElement()
    while (i < paramters.length) {
      if (!storage.eq(paramters(i).storage())) {
        return null
      }
      length += paramters(i).nElement()
      i += 1
    }

    if (length != storage.array().length) {
      return null
    }

    return Tensor(storage)
  }

  def quantize[@specialized(Float) T: ClassTag](model: AbstractModule[Activity, Activity, T])(
    implicit ev: TensorNumeric[T]): Module[T] = {
    type ModuleNode[T] = AbstractModule[Activity, Tensor[T], T]

    def convertGraph(model: Graph[T]): Graph[T] = {
      implicit def moduleToModuleNode(module: Module[T]): ModuleNode[T] =
        module.asInstanceOf[ModuleNode[T]]

      def replaceRef(node: Node[ModuleNode[T]],
        newNode: Node[ModuleNode[T]], refs: Seq[Node[ModuleNode[T]]]): Unit = {
        val buffer = refs.asInstanceOf[ArrayBuffer[Node[ModuleNode[T]]]]
        refs.zipWithIndex.filter(_._1 == node).foreach { x =>
          buffer.update(x._2, newNode)
        }
      }

      def replace(node: Node[ModuleNode[T]], module: ModuleNode[T],
        list: Array[Node[ModuleNode[T]]], index: Int): Unit = {
        // create a new node
        val newNode = Node(module)
        newNode.nextNodes.asInstanceOf[ArrayBuffer[Node[ModuleNode[T]]]] ++= node.nextNodes
        newNode.prevNodes.asInstanceOf[ArrayBuffer[Node[ModuleNode[T]]]] ++= node.prevNodes

        // prev.next
        newNode.prevNodes.foreach(n => replaceRef(node, newNode, n.nextNodes))

        // next.prev
        newNode.nextNodes.foreach(n => replaceRef(node, newNode, n.prevNodes))

        // update the list
        list.update(index, newNode)
      }

      val sortedNodes = model.backGraph.topologySort

      for (i <- sortedNodes.indices) {
        val node = sortedNodes(i)
        val module = node.element
        val waitedModule = substitute(module)

        if (waitedModule != module) {
          replace(node, waitedModule, sortedNodes, i)
        }
      }

      val inputs = sortedNodes.filter(n => n.prevNodes.isEmpty)
      val outputs = sortedNodes.filter(n => n.nextNodes.isEmpty)

      // create a new Graph, much simpler than replacing others in the old graph
      Graph(inputs, outputs)
    }

    def substitute(model: Module[T]): Module[T] = {
      model match {
        case container: Container[Activity, Activity, T] =>
          container match {
            case graph: Graph[T] =>
              convertGraph(graph)
            case _ =>
              // do with container
              for (i <- container.modules.indices) {
                container.modules(i) = substitute(container.modules(i))
              }
              container
          }
        case normalConv if normalConv.isInstanceOf[SpatialConvolution[T]] =>
          // do with normal convolution
          val conv = normalConv.asInstanceOf[SpatialConvolution[T]]
          val quantizedConv = new fixpoint.SpatialConvolution[T](
            conv.nInputPlane,
            conv.nOutputPlane,
            conv.kernelW,
            conv.kernelH,
            conv.strideW,
            conv.strideH,
            conv.padW,
            conv.padH,
            conv.nGroup)
          quantizedConv.initWeightAndBias(conv.weight, conv.bias)
        case dilatedConv if dilatedConv.isInstanceOf[SpatialDilatedConvolution[T]] =>
          // do with dilated convolution
          val conv = dilatedConv.asInstanceOf[SpatialDilatedConvolution[T]]
          val quantizedConv = new fixpoint.SpatialDilatedConvolution[T](
            conv.nInputPlane,
            conv.nOutputPlane,
            conv.kW,
            conv.kH,
            conv.dW,
            conv.dH,
            conv.padW,
            conv.padH,
            conv.dilationW,
            conv.dilationH)
          quantizedConv.initWeightAndBias(conv.weight, conv.bias)
        case normalLinear if normalLinear.isInstanceOf[Linear[T]] =>
          // do with linear
          val linear = normalLinear.asInstanceOf[Linear[T]]

          val quantizedLinear = new fixpoint.Linear[T](
            linear.weight.size(2),
            linear.weight.size(1))

          quantizedLinear.initWeightAndBias(linear.weight, linear.bias)
        case _ => model
      }
    }

    def reorganizeParameters(parameters: Array[Tensor[T]]): Tensor[T] = {
      var length = 0
      for (i <- parameters.indices) {
        if (parameters(i) != null) {
          length += parameters(i).nElement()
        }
      }

      val result = Tensor[T](length)

      var offset = 0
      for (i <- parameters.indices) {
        val parameter = parameters(i)

        if (parameter != null) {
          val length = parameter.nElement()

          val (src, srcOffset) = (parameter.storage().array(), parameter.storageOffset() - 1)
          val (dst, dstOffset) = (result.storage().array(), offset)

          val (size, stride) = (parameter.size(), parameter.stride())

          System.arraycopy(src, srcOffset, dst, dstOffset, length)
          parameter.set(result.storage(), offset + 1, size, stride)

          offset += length
        }
      }

      result
    }

    // deep copy a new model then substitute with all quantized version models
    val quantizedModel = substitute(model.cloneModule())

    val paras = quantizedModel.parameters()._1
    reorganizeParameters(paras)

    quantizedModel
  }
}
