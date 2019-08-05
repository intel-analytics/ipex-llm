/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api

import java.nio.ByteOrder

import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.nn.{Container, Graph, InitializationMethod}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.File
import com.intel.analytics.zoo.models.caffe.CaffeLoader
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.bigdl.utils.tf.{Session, TensorflowLoader}
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras.layers.WordEmbedding
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.net.{GraphNet, NetUtils}

import scala.reflect.ClassTag

/**
 * A placeholder to add layer's utilities
 */
trait Net {

  def isFrozen[T: ClassTag](): Boolean = {
    val labor = this.asInstanceOf[KerasLayer[Activity, Activity, T]].labor
    (labor.getScaleW() == 0) && (labor.getScaleB() == 0)
  }

  /**
   * Build graph: some other modules point to current module
   * @param vars upstream variables
   * @return Variable containing current module
   */
  def from[T: ClassTag](vars : Variable[T]*)(implicit ev: TensorNumeric[T]): Variable[T] = {
    new Variable(
      this.asInstanceOf[AbstractModule[Activity, Activity, T]].inputs(vars.map(_.node): _*))
  }
}

object Net {
  Model
  Sequential
  GraphNet
  WordEmbedding
  def setInitMethod(module: AbstractModule[_, _, _],
      weightInitMethod: InitializationMethod = null,
      biasInitMethod: InitializationMethod = null, throwException: Boolean = true): Unit = {
    module match {
      case i: Initializable =>
        i.setInitMethod(weightInitMethod, biasInitMethod)
      case k: KerasLayer[_, _, _] =>
        setInitMethod(k.labor, weightInitMethod, biasInitMethod, throwException)
      case c: Container[_, _, _] => // Some KerasLayer may be constructed by multiple layers
        c.modules.map { module =>
          setInitMethod(module, weightInitMethod, biasInitMethod, false)
        }
      case _ =>
        if (throwException) {
          throw new RuntimeException(s"$module does not support setInitMethod")
        }
    }
  }
  /**
   * Load model from path.
   *
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param weightPath : where weight is stored
   * @tparam T numeric type
   * @return An Analytics Zoo model.
   */
  def load[T: ClassTag](path : String,
      weightPath : String = null)(implicit ev: TensorNumeric[T])
  : KerasNet[T] = {
    val model = ModuleLoader.loadFromFile(path, weightPath)
    if (!model.isInstanceOf[KerasNet[T]]) {
      throw new RuntimeException(
        "Not an Analytics Zoo Keras-style model. Please use loadBigDL, loadCaffe or loadTF instead")
    }
    model.asInstanceOf[KerasNet[T]]
  }

  /**
   * Load BigDL model from path.
   *
   * @param path path to save module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @param weightPath : where weight is stored
   * @tparam T numeric type
   * @return model loaded from path
   */
  def loadBigDL[T: ClassTag](path : String,
      weightPath : String = null)(implicit ev: TensorNumeric[T])
  : GraphNet[T] = {
    val graph = ModuleLoader.loadFromFile(path, weightPath).toGraph()
    new GraphNet(graph)
  }

  /**
   * Load Torch model from path.
   *
   * @param path path to load module, local file system, HDFS and Amazon S3 is supported.
   *             HDFS path should be like "hdfs://[host]:[port]/xxx"
   *             Amazon S3 path should be like "s3a://bucket/xxx"
   * @tparam T numeric type
   * @return model loaded from path
   */
  def loadTorch[T: ClassTag](path : String)(implicit ev: TensorNumeric[T]):
  GraphNet[T] = {
    val graph = File.loadTorch[AbstractModule[Activity, Activity, T]](path).toGraph()
    new GraphNet[T](graph)
  }

  /**
   * Loaf caffe trained model from prototxt and weight files
   * @param defPath  caffe model definition file path
   * @param modelPath caffe model binary file containing weight and bias
   */
  def loadCaffe[T: ClassTag](defPath: String, modelPath: String)(
      implicit ev: TensorNumeric[T]): GraphNet[T] = {
    val graph = CaffeLoader.loadCaffe[T](defPath, modelPath)._1
      .asInstanceOf[Graph[T]]
    new GraphNet[T](graph)
  }

  /**
   * Load tensorflow model from its saved protobuf file.
   * @param graphFile where is the protobuf model file
   * @param inputs input node names
   * @param outputs output node names, the output tensor order is same with the node order
   * @param byteOrder byte order in the tensorflow file. The default value is little endian
   * @param binFile where is the model variable file
   * @return model loaded from path
   */
  def loadTF[T: ClassTag](graphFile: String, inputs: Seq[String], outputs: Seq[String],
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN,
      binFile: Option[String] = None)(
      implicit ev: TensorNumeric[T]): GraphNet[T] = {

    val graph = TensorflowLoader.load(graphFile, inputs, outputs, byteOrder, binFile)
      .asInstanceOf[Graph[T]]
    new GraphNet[T](graph)
  }

  /**
   * Load TensorFlow model from exported folder.
   * @param folder The folder path which contains 'frozen_inference_graph.pb' and
   *               'graph_meta.json'.
   * @return model loaded from path
   */
  def loadTF[T: ClassTag](folder: String)
      (implicit ev: TensorNumeric[T]): GraphNet[T] = {
    val (model, meta) = NetUtils.processTFFolder(folder)
    loadTF[T](model, NetUtils.removePort(meta.inputNames), NetUtils.removePort(meta.outputNames))
  }

  /**
   * Load tensorflow checkpoints
   * @param graphFile
   * @param binFile
   * @tparam T
   * @return
   */
  def loadTFCheckpoints[T: ClassTag](graphFile: String, binFile: String,
      byteOrder: ByteOrder = ByteOrder.LITTLE_ENDIAN)(
      implicit ev: TensorNumeric[T]): Session[T] = {
    TensorflowLoader.checkpoints(graphFile, binFile, byteOrder)
  }
}
