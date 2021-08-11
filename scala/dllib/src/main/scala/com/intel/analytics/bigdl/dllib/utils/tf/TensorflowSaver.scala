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

import java.io.{FileOutputStream, OutputStream}
import java.nio.ByteOrder

import com.google.protobuf.CodedOutputStream
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{File, FileWriter, T}
import org.apache.log4j.Logger
import org.tensorflow.framework._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.utils.tf.Tensorflow._

import scala.reflect.ClassTag

object TensorflowSaver {
  /**
   * Save a graph model to protobuf files so that it can be used in tensorflow inference.
   *
   * When save the model, placeholders will be added to the tf model as input nodes. So you need to
   * pass in the names and shape for the placeholders. BigDL model doesn't have such information.
   * The order of the placeholder information should be same as the inputs of the graph model
   *
   * @param model graph model instance
   * @param inputs input node defs
   * @param path where to save
   * @param byteOrder model byte order
   * @tparam T
   */
  def saveGraphWithNodeDef[T](
      model : Graph[T],
      inputs : Seq[NodeDef],
      path: String,
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
      extraNodes: Set[NodeDef] = Set()): Unit = {
    val inputNodeCache =
      new mutable.HashMap[String, ArrayBuffer[NodeDef]]()
    model.inputs.zip(inputs).foreach(n => {
      inputNodeCache(n._1.element.getName()) = ArrayBuffer(n._2)
    })

    val graphBuilder = GraphDef.newBuilder()
    inputs.foreach(graphBuilder.addNode(_))

    model.getSortedForwardExecutions.foreach(n => {
      val nodeDefs = maps(n.element.getClass.getName).toTFDef(n.element,
        inputNodeCache(n.element.getName()),
        byteOrder)
      nodeDefs.foreach(nDef => {
        graphBuilder.addNode(nDef)
      })
      n.nextNodes.foreach(n => {
        val list = inputNodeCache.getOrElse(n.element.getName(), ArrayBuffer())
        list.append(nodeDefs(0))
        inputNodeCache(n.element.getName()) = list
      })
    })

    extraNodes.foreach(graphBuilder.addNode(_))

    // Save to file
    var fw: FileWriter = null
    var out: OutputStream = null
    try {
      fw = FileWriter(path)
      out = fw.create(true)
      val output = CodedOutputStream.newInstance(out)
      val graph = graphBuilder.build()
      logger.debug("Graph definition is:")
      logger.debug(graph.toString)
      graph.writeTo(output)
      output.flush()
      out.flush()
      logger.info(s"Save as tensorflow model file to $path")
    } finally {
      if (out != null) out.close()
      if (fw != null) fw.close()
    }

  }

  /**
   * Save a graph model to protobuf files so that it can be used in tensorflow inference.
   *
   * When save the model, placeholders will be added to the tf model as input nodes. So you need to
   * pass in the names and shape for the placeholders. BigDL model doesn't have such information.
   * The order of the placeholder information should be same as the inputs of the graph model
   *
   * @param model graph model instance
   * @param inputs placeholder information
   * @param path where to save
   * @param byteOrder model byte order
   * @param dataFormat model data format
   * @tparam T
   */
  def saveGraph[T: ClassTag](
      model : Graph[T],
      inputs : Seq[(String, Seq[Int])],
      path: String,
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
      dataFormat: TensorflowDataFormat = TensorflowDataFormat.NHWC)(
    implicit ev: TensorNumeric[T]): Unit = {
    // Check if there's pooling layer in which ceilMode is enable and pad is zero, we need double
    // check if the ceilMode is real needed
    val ceiledPoolingModules = model.modules.filter(m =>
      if (m.isInstanceOf[SpatialMaxPooling[_]]) {
        val a = m.asInstanceOf[SpatialMaxPooling[_]]
        a.ceilMode == true && a.padH == 0 && a.padW == 0
      } else if (m.isInstanceOf[SpatialAveragePooling[_]]) {
        val a = m.asInstanceOf[SpatialAveragePooling[_]]
        a.ceilMode == true && a.padH == 0 && a.padW == 0
      } else {
        false
      })

    if (ceiledPoolingModules.size != 0) {
      val inputTensors = inputs.map(shape => Tensor[T]().resize(shape._2.toArray))
      val inputActivity = if (inputTensors.size == 1) {
        inputTensors.head
      } else {
        val t = T()
        var i = 1
        inputTensors.foreach(tensor => {
          t(i) = tensor
          i += 1
        })
        t
      }
      model.forward(inputActivity)
    }


    val inputNodeDefs = inputs.map(input =>
      placeholder(model.getNumericType(), input._2, input._1)
    )
    saveGraphWithNodeDef(model, inputNodeDefs, path, byteOrder)
  }

  /**
   * Register a customized BigDL module saver.
   * @param className class name of the BigDL module
   * @param saver customized saver
   */
  def register(className : String, saver: BigDLToTensorflow): Unit = {
    maps(className) = saver
  }

  private val logger = Logger.getLogger(getClass)

  private val maps = mutable.Map[String, BigDLToTensorflow](
    getNameFromObj(TemporalConvolution.getClass.getName) -> TemporalConvolutionToTF,
    getNameFromObj(ReLU.getClass.getName) -> ReLUToTF,
    getNameFromObj(Linear.getClass.getName) -> LinearToTF,
    getNameFromObj(SpatialConvolution.getClass.getName) -> SpatialConvolutionToTF,
    getNameFromObj(Squeeze.getClass.getName) -> SqueezeToTF,
    getNameFromObj(Tanh.getClass.getName) -> TanhToTF,
    getNameFromObj(Reshape.getClass.getName) -> ReshapeToTF,
    getNameFromObj(View.getClass.getName) -> ViewToTF,
    getNameFromObj(SpatialMaxPooling.getClass.getName) -> MaxpoolToTF,
    getNameFromObj(Padding.getClass.getName) -> PaddingToTF,
    getNameFromObj(SpatialAveragePooling.getClass.getName) -> AvgpoolToTF,
    getNameFromObj(Sigmoid.getClass.getName) -> SigmoidToTF,
    getNameFromObj(Dropout.getClass.getName) -> DropoutToTF,
    getNameFromObj(CAddTable.getClass.getName) -> CAddTableToTF,
    getNameFromObj(CMulTable.getClass.getName) -> CMultTableToTF,
    getNameFromObj(JoinTable.getClass.getName) -> JoinTableToTF,
    getNameFromObj(Mean.getClass.getName) -> MeanToTF,
    getNameFromObj(SoftMax.getClass.getName) -> SoftMaxToTF,
    getNameFromObj(LogSoftMax.getClass.getName) -> LogSoftMaxToTF,
    getNameFromObj(SpatialBatchNormalization.getClass.getName) -> BatchNorm2DToTF,
    getNameFromObj(Input.getClass.getName) -> InputToTF,
    getNameFromObj(Sigmoid.getClass.getName) -> SigmoidToTF,
    getNameFromObj(Scale.getClass.getName) -> ScaleToTF,
    getNameFromObj(SpatialCrossMapLRN.getClass.getName) -> LRNToTF
  )

  private def getNameFromObj(name: String) : String = name.substring(0, name.length - 1)
}

