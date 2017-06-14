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

import java.io.FileOutputStream
import java.nio.ByteOrder

import com.google.protobuf.CodedOutputStream
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger
import org.tensorflow.framework._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.utils.tf.Tensorflow._

object TensorflowSaver {
  /**
   * Save a graph model to protobuf files so that it can be used in tensorflow inference.
   *
   * When save the model, placeholders will be added to the tf model as input nodes. So you need to
   * pass in the names and shape for the placeholders. BigDL model doesn't have such information.
   * The order of the placeholde information should be same as the inputs of the graph model
   *
   * @param model graph model instance
   * @param inputs input node defs
   * @param path where to save
   * @param byteOrder model byte order
   * @tparam T
   */
  def saveGraphWitNodeDef[T](
      model : Graph[T],
      inputs : Seq[NodeDef],
      path: String,
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
      extraNodes: Set[NodeDef] = Set()): Unit = {
    val inputNodeCache =
      new mutable.HashMap[AbstractModule[Activity, Tensor[T], T], ArrayBuffer[NodeDef]]()
    model.inputs.zip(inputs).foreach(n => {
      inputNodeCache(n._1.element) = ArrayBuffer(n._2)
      println()
    })

    val graphBuilder = GraphDef.newBuilder()
    inputs.foreach(graphBuilder.addNode(_))

    model.executions.foreach(n => {
      val nodeDefs = maps(n.element.getClass.getName).toTFDef(n.element, inputNodeCache(n.element),
        byteOrder)
      nodeDefs.foreach(nDef => {
        graphBuilder.addNode(nDef)
      })
      n.nextNodes.foreach(n => {
        val list = inputNodeCache.getOrElse(n.element, ArrayBuffer())
        list.append(nodeDefs(0))
        inputNodeCache(n.element) = list
      })
    })

    extraNodes.foreach(graphBuilder.addNode(_))

    // Save to file
    val os = new FileOutputStream(path)
    val output = CodedOutputStream.newInstance(os)
    val graph = graphBuilder.build()
    logger.info("Graph definition is:")
    logger.info(graph.toString)
    graph.writeTo(output)
    output.flush()
    os.close()
    logger.info(s"Save as tensorflow model file to $path")
  }

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
   * @param byteOrder model byte order
   * @param dataFormat model data format
   * @tparam T
   */
  def saveGraph[T](
      model : Graph[T],
      inputs : Seq[(String, Seq[Int])],
      path: String,
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
      dataFormat: TensorflowDataFormat = TensorflowDataFormat.NHWC): Unit = {
    val inputNodeDefs = inputs.map(input =>
      placeholder(model.getNumericType(), input._2, input._1)
    )
    saveGraphWitNodeDef(model, inputNodeDefs, path, byteOrder)
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
    getNameFromObj(Sigmoid.getClass.getName) -> SigmoidToTF
  )

  private def getNameFromObj(name: String) : String = name.substring(0, name.length - 1)
}

