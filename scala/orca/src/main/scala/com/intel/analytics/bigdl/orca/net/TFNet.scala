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
package com.intel.analytics.zoo.pipeline.api.net

import java.io.{File, FileInputStream, InputStream}
import java.nio._

import com.esotericsoftware.kryo.io.{Input, Output}
import com.esotericsoftware.kryo.{Kryo, KryoSerializable}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, T}
import com.intel.analytics.zoo.pipeline.api.net.TFNet.TFGraphHolder
import org.apache.spark.utils.SparkUtils
import org.tensorflow.framework.GraphDef
import org.tensorflow.types.UInt8
import org.tensorflow.{DataType, Graph, Session, Tensor => TTensor}

import scala.collection.JavaConverters._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.slf4j.LoggerFactory

import scala.collection.mutable

/**
 * [[TFNet]] wraps a tensorflow subgraph as a layer, and use tensorflow to
 * calculate the layer's output.
 *
 * This subgraph should not contain any tensorflow Variable and the input/output
 * must be numeric types
 *
 * When used with other layers for training, there should be no trainable layer
 * before this one, as the gradInput of this layer is always zero.
 *
 * @param graphDef serialized representation of a graph
 * @param inputNames the input tensor names of this subgraph
 * @param outputNames the output tensor names of this subgraph
 */
class TFNet private(graphDef: TFGraphHolder,
            val inputNames: Seq[String],
            val outputNames: Seq[String],
                    config: Array[Byte])
  extends AbstractModule[Activity, Activity, Float] {

  private def graph = graphDef.tfGraph

  output = {
    if (outputNames.length == 1) {
      Tensor[Float]()
    } else {
      val t = T()
      var i = 0
      while (i < outputNames.length) {
        t.insert(Tensor[Float]())
        i = i + 1
      }
      t
    }
  }

  gradInput = {
    if (inputNames.length == 1) {
      Tensor[Float]()
    } else {
      val t = T()
      var i = 0
      while (i < inputNames.length) {
        t.insert(Tensor[Float]())
        i = i + 1
      }
      t
    }
  }

  private def getOutput(idx: Int): Tensor[Float] = {
    if (output.isTable) {
      output.toTable[Tensor[Float]](idx)
    } else {
      output.toTensor[Float]
    }
  }

  @transient
  private lazy val sess = {
    val sess = new Session(graphDef.tfGraph, config)
    sess
  }

  private val inputTypes = inputNames.map { name =>
    val Array(op, idx) = name.split(":")
    val operation = graph.operation(op)
    val output = operation.output(idx.toInt)
    output.dataType()
  }

  // add Cast Operation if the output tensor is not of type Float
  private val floatOutputNames = outputNames.map { name =>
    val Array(op, idx) = name.split(":")
    val operation = graph.operation(op)
    val output = operation.output(idx.toInt)
    if (output.dataType() != DataType.FLOAT) {
      val name = graph.opBuilder("Cast", s"${op}_to_float")
        .addInput(output)
        .setAttr("DstT", DataType.FLOAT)
        .setAttr("SrcT", output.dataType())
        .build()
        .name()
      s"$name:0"
    } else {
      name
    }
  }

  private def getShape(names: Seq[String]) = {
    val shapes = names.map { name =>
      val Array(op, idx) = name.split(":")
      val shape = graph.operation(op).output(idx.toInt).shape()
      Shape((0 until shape.numDimensions()).map(shape.size(_).toInt).toArray)
    }

    if (shapes.length == 1) {
      shapes.head
    } else {
      MultiShape(shapes.toList)
    }
  }

  override def parameters(): (Array[Tensor[Float]], Array[Tensor[Float]]) = {
    (Array(), Array())
  }

  private def bigdl2Tf(t: Tensor[Float], dataType: DataType): TTensor[_] = {

    require(t.isContiguous(), "input to tfnet must be contiguous")
    val shape = t.size().map(_.toLong)
    val arr = t.storage().array()
    val offset: Int = t.storageOffset() - 1
    val length: Int = shape.product.toInt

    if (dataType == DataType.FLOAT) {
      val buffer = FloatBuffer.wrap(arr, offset, length)
      TTensor.create(shape, buffer)
    } else if (dataType == DataType.UINT8) {
      val buffer = ByteBuffer.wrap(TFNet.floatToUint8(arr), offset, length)
      TTensor.create(classOf[UInt8], shape, buffer)
    } else if (dataType == DataType.INT32) {
      val buffer = IntBuffer.wrap(TFNet.floatToInt(arr), offset, length)
      TTensor.create(shape, buffer)
    } else if (dataType == DataType.INT64) {
      val buffer = LongBuffer.wrap(TFNet.floatToLong(arr), offset, length)
      TTensor.create(shape, buffer)
    } else if (dataType == DataType.DOUBLE) {
      val buffer = DoubleBuffer.wrap(TFNet.floatToDouble(arr), offset, length)
      TTensor.create(shape, buffer)
    } else {
      throw new Exception(s"data type ${dataType} are not supported")
    }

  }

  private def tf2bigdl(t: TTensor[Float], output: Tensor[Float]) = {
    val shape = t.shape().map(_.toInt)
    output.resize(shape)
    val buffer = FloatBuffer.wrap(
      output.storage().array(),
      output.storageOffset() - 1,
      shape.product)
    t.writeTo(buffer)
  }

  override def updateOutput(input: Activity): Activity = {
    val data = if (input.isTensor) {
      require(inputTypes.size == 1,
        s"This TFNet requires ${inputTypes.size} inputs, which are $inputNames," +
          s" but only one input is provided")
      val tfTensor = bigdl2Tf(input.toTensor[Float], inputTypes.head)
      Seq(tfTensor)
    } else {
      val t = input.toTable
      require(inputTypes.size == t.length(),
        s"This TFNet requires ${inputTypes.size} inputs, which are $inputNames," +
          s" but ${t.length()} inputs are provided")
      for (i <- 1 to t.length()) yield {
        bigdl2Tf(t[Tensor[Float]](i), inputTypes(i-1))
      }
    }

    val runner = sess.runner()
    floatOutputNames.foreach(runner.fetch)
    inputNames.zipWithIndex.foreach { case (name, idx) =>
      runner.feed(name, data(idx))
    }

    val outputs = runner.run()
    outputs.asScala.zipWithIndex.foreach { case (t, idx) =>
      tf2bigdl(t.asInstanceOf[TTensor[Float]], getOutput(idx + 1))
    }
    // clean up resources
    data.foreach(_.close())
    outputs.asScala.foreach(_.close())
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    if (gradInput.isTable) {
      var i = 0
      while (i < gradInput.toTable.length()) {
        gradInput.toTable[Tensor[Float]](i + 1)
          .resizeAs(input.toTable[Tensor[Float]](i + 1))
        i = i + 1
      }
    } else {
      gradInput.toTensor[Float]
        .resizeAs(input.toTensor[Float])
    }

    gradInput
  }
}

object TFNet {

  private def isDriver = try {
    SparkUtils.isDriver
  } catch {
    case e: NullPointerException =>
      true
  }

  private def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    logger.debug(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  class TFGraphHolder(@transient var tfGraph: Graph, private var id: String)
    extends Serializable with KryoSerializable {

    private trait CommonOutputStream {
      def writeInt(value: Int): Unit
      def write(value: Array[Byte]): Unit
      def writeString(value: String): Unit
    }

    private trait CommonInputStream {
      def readInt(): Int
      def read(buff: Array[Byte], off: Int, len: Int): Int
      def skip(len: Int): Unit
      def readString(): String
    }

    private def writeInternal(out: CommonOutputStream): Unit = {

      val (graphDef, _) = getOrCreateGraphDef(id) {
        timing("export as graph def") {
          tfGraph.toGraphDef
        }
      }
      val len = graphDef.length
      out.writeString(id)
      if (isDriver) {
        out.writeInt(len)
        timing(s"writing ${len / 1024 / 1024}Mb graph def to stream") {
          out.write(graphDef)
        }
      } else {
        out.writeInt(0)
      }
    }

    private def readInternal(in: CommonInputStream): Unit = {
      id = in.readString()
      val (graphDef, graphDefIsCreated) = getOrCreateGraphDef(id) {
        val len = in.readInt()
        require(len != 0, "GraphDef length should not be zero," +
          "please set logging level to debug for more information")
        val graphDef = new Array[Byte](len)
        timing("reading graph def from stream") {
          var numOfBytes = 0
          while (numOfBytes < len) {
            val read = in.read(graphDef, numOfBytes, len - numOfBytes)
            numOfBytes += read
          }
        }
        graphDef
      }

      if (!graphDefIsCreated) {
        val len = in.readInt()
        in.skip(len)
      }

      val (graph, _) = getOrCreateGraph(id) {
        timing("creating graph obj from graph def") {
          val g = new Graph()
          g.importGraphDef(graphDef)
          g
        }

      }
      tfGraph = graph
    }

    private def writeObject(out: java.io.ObjectOutputStream): Unit = {
      writeInternal(new CommonOutputStream {
        override def writeInt(value: Int): Unit = out.writeInt(value)

        override def write(value: Array[Byte]): Unit = out.write(value)

        override def writeString(str: String): Unit = out.writeUTF(str)
      })
    }

    private def readObject(in: java.io.ObjectInputStream): Unit = {
      readInternal(new CommonInputStream {
        override def read(buff: Array[Byte], off: Int, len: Int): Int = in.read(buff, off, len)

        override def skip(len: Int): Unit = in.skip(len)

        override def readInt(): Int = in.readInt()

        override def readString(): String = in.readUTF()
      })
    }

    override def read(kryo: Kryo, in: Input): Unit = {
      readInternal(new CommonInputStream {
        override def read(buff: Array[Byte], off: Int, len: Int): Int = in.read(buff, off, len)

        override def skip(len: Int): Unit = in.skip(len)

        override def readInt(): Int = in.readInt()

        override def readString(): String = in.readString()
      })
    }

    override def write(kryo: Kryo, out: Output): Unit = {
      writeInternal(new CommonOutputStream {
        override def writeInt(value: Int): Unit = out.writeInt(value)

        override def write(value: Array[Byte]): Unit = out.write(value)

        override def writeString(value: String): Unit = out.writeString(value)
      })
    }
  }

  private val logger = LoggerFactory.getLogger(getClass)

  private val graphRegistry = new mutable.WeakHashMap[String, Graph]()

  private val graphDefRegistry = new mutable.WeakHashMap[String, Array[Byte]]()

  private[zoo] def getGraphRegistrySize = graphDefRegistry.size

  private[zoo] def getGraphDefRegistrySize = graphDefRegistry.size

  private def getOrCreateGraphDef(id: String)
                                 (createGraphDef: => Array[Byte]): (Array[Byte], Boolean) = {
    if (graphDefRegistry.contains(id)) {
      logger.debug(s"graphDef: $id already exist, read from registry. " +
        s"Registry size: $getGraphDefRegistrySize")
      (graphDefRegistry(id), false)
    } else {
      this.synchronized {
        if (graphDefRegistry.contains(id)) {
          logger.debug(s"graphDef: $id already exist, read from registry. " +
            s"Registry size: $getGraphDefRegistrySize")
          (graphDefRegistry(id), false)
        } else {
          val graphDef = createGraphDef
          graphDefRegistry.put(id, graphDef)
          logger.debug(s"graphDef: $id does not exist, create it. " +
            s"Registry size: $getGraphDefRegistrySize")
          (graphDef, true)
        }
      }
    }
  }

  private def getOrCreateGraph(id: String)
                                 (createGraph: => Graph): (Graph, Boolean) = {
    if (graphRegistry.contains(id)) {
      logger.debug(s"graph: $id already exist, read from registry. " +
        s"Registry size: $getGraphRegistrySize")
      (graphRegistry(id), false)
    } else {
      this.synchronized {
        if (graphRegistry.contains(id)) {
          logger.debug(s"graph: $id already exist, read from registry. " +
            s"Registry size: $getGraphRegistrySize")
          (graphRegistry(id), false)
        } else {
          val graph = createGraph
          graphRegistry.put(id, graph)
          logger.debug(s"graph: $id does not exist, create it. " +
            s"Registry size: $getGraphRegistrySize")
          (graph, true)
        }
      }
    }
  }

  implicit val formats = DefaultFormats

  val defaultSessionConfig = Seq(16, 1, 40, 1, 72, 1).map(_.toByte).toArray
  // Ideally we should use the following code, however, importing tensorflow proto
  // will conflict with bigdl.

//  val defaultSessionConfig = ConfigProto.newBuilder()
//    .setInterOpParallelismThreads(1)
//    .setIntraOpParallelismThreads(1)
//    .setUsePerSessionThreads(true)
//    .build().toByteArray

  private def floatToInt(array: Array[Float]): Array[Int] = {
    val result = new Array[Int](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toInt
      i = i + 1
    }
    result
  }

  private def floatToLong(array: Array[Float]): Array[Long] = {
    val result = new Array[Long](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toLong
      i = i + 1
    }
    result
  }

  private def floatToDouble(array: Array[Float]): Array[Double] = {
    val result = new Array[Double](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toDouble
      i = i + 1
    }
    result
  }

  private def floatToUint8(array: Array[Float]): Array[Byte] = {
    val result = new Array[Byte](array.length)
    var i = 0
    while (i < array.length) {
      result(i) = array(i).toByte
      i = i + 1
    }
    result
  }

  /**
   * Create a TFNet
   * @param graphDef the tensorflow GraphDef object
   * @param inputNames the input tensor names of this subgraph
   * @param outputNames the output tensor names of this subgraph
   * @return
   */
  private def apply(graphDef: GraphDef, graphId: String, inputNames: Seq[String],
            outputNames: Seq[String], config: Array[Byte] = defaultSessionConfig): TFNet = {
    val graph = new Graph()
    graph.importGraphDef(graphDef.toByteArray)

    new TFNet(new TFGraphHolder(graph, graphId), inputNames, outputNames, config)
  }

  /**
   * Create a TFNet
   * @param path the file path of a graphDef
   * @param inputNames the input tensor names of this subgraph
   * @param outputNames the output tensor names of this subgraph
   * @return
   */
  def apply(path: String,
            inputNames: Seq[String],
            outputNames: Seq[String], config: Array[Byte]): TFNet = {
    val graphDef = parseGraph(path)
    TFNet(graphDef, path, inputNames, outputNames, config)
  }

  /**
   * Create a TFNet
   * @param path the file path of a graphDef
   * @param inputNames the input tensor names of this subgraph
   * @param outputNames the output tensor names of this subgraph
   * @return
   */
  def apply(path: String,
            inputNames: Seq[String],
            outputNames: Seq[String]): TFNet = {
    val graphDef = parseGraph(path)
    TFNet(graphDef, path, inputNames, outputNames, defaultSessionConfig)
  }


  def apply(folder: String): TFNet = {
    val (model, inputs, outputs) = NetUtils.processTFFolder(folder)
    TFNet(model, inputs, outputs, defaultSessionConfig)
  }

  private def parseGraph(graphProtoTxt: String) : GraphDef = {
    var fr: File = null
    var in: InputStream = null
    try {
      fr = new File(graphProtoTxt)
      in = new FileInputStream(fr)

      GraphDef.parseFrom(in)
    } finally {
      if (in != null) in.close()
    }
  }
}
