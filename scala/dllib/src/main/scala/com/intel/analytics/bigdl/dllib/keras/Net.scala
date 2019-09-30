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

import java.io.{BufferedReader, BufferedWriter, FileOutputStream, FileWriter, InputStreamReader, File => JFile}
import java.nio.ByteOrder
import java.util

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable}
import com.intel.analytics.bigdl.nn.keras.{KerasIdentityWrapper, KerasLayer}
import com.intel.analytics.bigdl.nn.{Container, Graph, InitializationMethod, StaticGraph, Identity => BIdentity, Sequential => TSequential}
import com.intel.analytics.bigdl.python.api.PythonBigDL
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{File, Shape}
import com.intel.analytics.zoo.models.caffe.CaffeLoader
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.bigdl.utils.tf.{Session, TensorflowLoader}
import com.intel.analytics.zoo.common.Utils
import com.intel.analytics.zoo.pipeline.api.autograd.Variable
import com.intel.analytics.zoo.pipeline.api.keras.layers.{KerasLayerWrapper, Merge, WordEmbedding}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.keras.models.{KerasNet, Model, Sequential}
import com.intel.analytics.zoo.pipeline.api.net.{GraphNet, NetUtils}
import org.apache.commons.io.FileUtils
import org.apache.log4j.Logger
import org.apache.spark.bigdl.api.python.BigDLSerDe
import org.apache.zookeeper.KeeperException.UnimplementedException

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

  private[zoo] def toKeras2(): String = {
    throw new UnimplementedException()
  }

  /**
   * Get keras-like weights.
   * Need to override this when this default weights doesn't match the weights in Keras.
   * @return keras-like weights.
   */
  private[zoo] def getKerasWeights(): Array[Tensor[Float]] = {
    if (this.asInstanceOf[AbstractModule[_, _, _]].parameters()._1.length != 0) {
      val weights = this.asInstanceOf[AbstractModule[_, _, _]].parameters()._1
      val kWeights = Array.tabulate(weights.length)(_ => Tensor[Float]())
      (0 until weights.length).foreach(i =>
        weights(i).cast[Float](kWeights(i).resizeAs(weights(i))))
      kWeights
    } else {
      Array()
    }
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

  private[zoo] def saveToKeras2[T: ClassTag](
        model: Net,
        filePath: String,
        python: String = "python")(implicit ev: TensorNumeric[T]): Unit = {
    NetSaver.saveToKeras2(model.asInstanceOf[Module[T]], filePath, python)
  }

  private[zoo] def saveToTf[T: ClassTag](
        model: Net,
        dir: String,
        python: String = "python")(implicit ev: TensorNumeric[T]): Unit = {
    NetSaver.saveToTf(model.asInstanceOf[Module[T]], dir, python)
  }

  private[zoo] def getName(name: String): String = {
    name.split("\\.").last
  }

  private[zoo] def inputShapeToString(
        inputShape: Shape,
        paramName: String = "input_shape"): Map[String, String] = {
    if (inputShape != null) {
      Map(paramName -> s"(${inputShape.toSingle().mkString(", ")},)")
    } else {
      Map()
    }
  }

  private[zoo] def arrayToString(array: Seq[Int], name: String): Map[String, String] = {
    Map(name -> s"(${array.mkString(", ")})")
  }

  private[zoo] def activationToString(
        activation: AbstractModule[_, _, _],
        paramName: String = "activation"): Map[String, String] = {
    val trueActivation = if (activation.isInstanceOf[KerasIdentityWrapper[_]]) {
      activation.asInstanceOf[KerasIdentityWrapper[_]].layer
    } else {
      activation
    }
    if (activation != null) {
      Map(paramName -> s"'${KerasUtils.getActivationName(trueActivation)}'")
    } else {
      Map()
    }

  }

  private[zoo] def param(
        boolean: Boolean,
        paramName: String): Map[String, String] = {
    Map(paramName -> s"${if (boolean) "True" else "False"}")
  }

  private[zoo] def param(
        integer: Int,
        paramName: String): Map[String, String] = {
    Map(paramName -> integer.toString)
  }

  private[zoo] def param(
        double: Double,
        paramName: String): Map[String, String] = {
    Map(paramName -> double.toString)
  }

  private[zoo] def param(
        name: String,
        paramName: String = "name"): Map[String, String] = {
    Map(paramName -> s"'$name'")
  }

  private[zoo] def kerasDef(
        module: Module[_],
        params: Map[String, String]): String = {
    s"${Net.getName(module.getClass.getName)}(" +
      params.map(v => s"${v._1}=${v._2}").mkString(", ") + ")"
  }

  private[zoo] def kerasDef(
       moduleType: String,
       params: Map[String, String]): String = {
    s"${moduleType}(" +
      params.map(v => s"${v._1}=${v._2}").mkString(", ") + ")"
  }

  protected object NetSaver {
    private val logger = Logger.getLogger(getClass)

    protected val header =
      """
        |from tensorflow.keras.models import Sequential, Model
        |from tensorflow.keras.layers import *
        |from pyspark.serializers import PickleSerializer
        |
        |def load_to_numpy(file):
        |    in_file = open(file, "rb")
        |    data = in_file.read()
        |    in_file.close()
        |    r=PickleSerializer().loads(data, encoding="bytes")
        |    return r.to_ndarray()
      """.stripMargin + "\n"

    protected val tfHeader =
      """
        |from zoo.util.tf import export_tf
        |from tensorflow.keras import backend as K
        |import tensorflow as tf
      """.stripMargin + "\n"

    def save[T: ClassTag](
          module: Module[T],
          path: String,
          python: String,
          saveCommand: String)
          (implicit ev: TensorNumeric[T]): Unit = {
      val tmpDir = Utils.createTmpDir("ZooKeras")
      logger.info(s"Write model's temp file to ${tmpDir}")
      val modelFile = tmpDir.toString + s"/${module.getName()}.py"
      val bw = new BufferedWriter(new FileWriter(modelFile))
      bw.write(header)
      module match {
        case s: Sequential[T] => export(s, bw)
        case m: Model[T] => export(m, bw)
        case _ =>
          throw new IllegalArgumentException(s"${module.getClass.getName} is not supported.")
      }
      bw.write(saveWeights(module, tmpDir.toString))
      bw.write(saveCommand)
      bw.flush()
      bw.close()
      execCommand(s"${python} ${modelFile}")
      FileUtils.deleteDirectory(tmpDir.toFile())
    }

    def saveToTf[T: ClassTag](m: Module[T], path: String, python: String)
                                 (implicit ev: TensorNumeric[T]): Unit = {
      val saveCommand = tfHeader +
        s"export_tf(K.get_session(), '${path}', model.inputs, model.outputs)\n"
      save(m, path, python, saveCommand)
    }

    def saveToKeras2[T: ClassTag](m: Module[T], path: String, python: String)
                      (implicit ev: TensorNumeric[T]): Unit = {
      save(m, path, python, s"model.save('$path')\n")
    }

    def execCommand(command: String): Unit = {
      val proc = Runtime.getRuntime().exec(command)
      proc.waitFor()
      if (proc.exitValue() != 0) {
        val error = new BufferedReader(new InputStreamReader(proc.getErrorStream()))
        val errorMsg = new StringBuilder()
        var line = error.readLine()
        while(line != null) {
          errorMsg.append(line + "\n")
          line = error.readLine()
        }
        throw new RuntimeException(s"Export Keras2 model failed:\n" + errorMsg.toString())
      }
    }

    def export[T: ClassTag](
          model: Model[T],
          writer: BufferedWriter): Unit = {
      val inputs = model.getInputs()
      val outputs = model.getOutputs()
      val nodes = model.labor.asInstanceOf[StaticGraph[T]].getSortedForwardExecutions()
      nodes.foreach(export(_, writer))
      val inputsName = inputs.map(_.element.getName).mkString(", ")
      val outputsName = outputs.map(_.element.getName).mkString(", ")
      writer.write(s"${model.getName()} = Model(inputs=[${inputsName}]," +
        s" outputs=[${outputsName}])\n")
    }

    def export[T: ClassTag](
          node: ModuleNode[T],
          writer: BufferedWriter): Unit = {
      val element = node.element
      if (!element.isInstanceOf[Net]) {
        throw new IllegalArgumentException(s"Unsupported layer ${element.getName()}")
      } else {
        val pre = if (node.prevNodes.length == 1) {
          s"(${node.prevNodes(0).element.getName})"
        } else if (node.prevNodes.length > 1) {
          s"([${node.prevNodes.map(_.element.getName).mkString(", ")}])"
        } else {
          ""
        }
        writer.write(s"${element.getName()} = ${element.asInstanceOf[Net].toKeras2()}${pre}\n")
        writer.flush()
      }
    }

    def export[T: ClassTag](
          sequential: Sequential[T],
          writer: BufferedWriter): Unit = {
      writer.write(s"${sequential.getName()} = " +
        s"Sequential(name='${(sequential.getName())}')\n")
      val modules = sequential.modules(0).asInstanceOf[TSequential[T]].modules
      modules.foreach{ module =>
        if (module.isInstanceOf[Sequential[T]]) {
          export(module.asInstanceOf[Sequential[T]], writer)
          writer.write(s"${sequential.getName()}.add(${module.getName})\n")
        } else if (module.isInstanceOf[Net]) {
          writer.write(s"${module.getName()} = ${module.asInstanceOf[Net].toKeras2()}\n")
          writer.write(s"${sequential.getName()}.add(${module.getName})\n")
        } else {
          throw new IllegalArgumentException(s"unkown type ${this.getClass.getName}")
        }
      }
    }

    private[zoo] def saveWeights[T: ClassTag](
                                                 module: AbstractModule[_, _, T], path: String)
                                             (implicit ev: TensorNumeric[T]): String = {
      val moduleName = module.getName()
      var i = 0
      val wStrings = module.asInstanceOf[Net].getKerasWeights().map{p =>
        val pName = s"${moduleName}_p${i}"
        val pPath = getUniqueFile(s"${path}/${pName}")
        saveToJTensor(p, pPath)
        i += 1
        (s"${pName} = load_to_numpy('${pPath}')",
          pName)
      }
      val loadWeights = wStrings.map(_._1).mkString("\n")
      val weightsList = wStrings.map(_._2).mkString(",")
      loadWeights + "\n" +
        s"${moduleName}.set_weights([${weightsList}])\n"
    }

    private def getUniqueFile(path: String): JFile = {
      var file = new JFile(path)
      var i = 0
      while(file.exists()) {
        file = new JFile(path + s".$i")
        i += 1
      }
      file
    }

    private def saveToJTensor[T: ClassTag](
                                              tensor: Tensor[T], file: JFile)
                                          (implicit ev: TensorNumeric[T]): Unit = {
      val pythonBigDL = new PythonBigDL[T]()
      val jt = pythonBigDL.toJTensor(tensor)
      val bytes = BigDLSerDe.dumps(jt)
      val fio = new FileOutputStream(file)
      fio.write(bytes)
      fio.flush()
      fio.close()
    }

  }
}
