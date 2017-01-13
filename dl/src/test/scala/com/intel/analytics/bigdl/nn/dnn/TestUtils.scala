/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.dnn

import java.io.{File, PrintWriter}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.{Files, Paths, StandardOpenOption}

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Container
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.sys.process._

object Tools {
  sealed trait Direction
  case object Forward extends Direction
  case object Backward extends Direction

  def error[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T])
                        (implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() == tensor2.nElement())
    var ret = 0.0
    val storage1 = tensor1.storage().array()
    val storage2 = tensor2.storage().array()
    for (i <- 0 until tensor1.nElement()) {
      ret += math.abs(
        ev.toType[Double](storage1(i)) - ev.toType[Double](storage2(i)))
    }
    ret
  }

  def cumulativeError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)
                                  (implicit ev: TensorNumeric[T]): Double = {
    val ret = error[T](tensor1, tensor2)
    println((msg, "CUMULATIVE ERROR:", ret).productIterator.mkString(" ").toUpperCase)
    ret
  }

  def averageError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)
                               (implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() > 0)
    val ret = error[T](tensor1, tensor2) / tensor1.nElement()
    println((msg, "AVERAGE ERROR:", ret).productIterator.mkString(" ").toUpperCase)
    ret
  }

  def averageError[T: ClassTag](m1: Map[String, Tensor[T]], m2: Map[String, Tensor[T]],
                                err: Map[String, Double])(implicit ev: TensorNumeric[T]): Unit = {
    require(m1.keySet == m2.keySet)
    require(m1.keySet subsetOf err.keySet)

    m1.keySet.foreach(i => {
      val err = error(m1(i), m2(i)) / m1(i).nElement()
      printf("%20s = %E\n", i.toUpperCase(), err)
    })
  }

  def averageAllTensors[T: ClassTag](tensor1: Tensor[T], msg: String = "Unknown")
                                    (implicit ev: TensorNumeric[T]): Unit = {
    val sum = tensor1.storage().array().foldLeft(ev.fromType[Int](0))((l, r) => ev.plus(l, r))
    val num = ev.fromType[Int](tensor1.nElement())
    println(("AVERGE", msg, ev.divide(sum, num)).productIterator.mkString(" ").toUpperCase())
  }

  /*
   * @brief read binary in tmp dir to Tensor, which is used for comparing
   *        with Intel Caffe with MKL-DNN
   */
  def loadTensor[T: ClassTag](name: String, size: Array[Int], identity: String)
                             (implicit ev: TensorNumeric[T]): Tensor[T] = {
    val tensor = Tensor[T]()
    val tmpdir = System.getProperty("java.io.tmpdir")
    val file = (if (tmpdir.endsWith("/")) tmpdir else tmpdir + "/", name,
      if (!identity.isEmpty) { "." + identity } else "", ".bin").productIterator.mkString("")

    if (Files.exists(Paths.get(file))) {
      ev.getType() match {
        case FloatType => setTensorFloat()
        case DoubleType => setTensorDouble()
        case _ => throw new UnsupportedOperationException(s"only support float and double")
      }

      def loadData(name: String): ByteBuffer = {
        val fileChannel: FileChannel =
          Files.newByteChannel(Paths.get(name), StandardOpenOption.READ).asInstanceOf[FileChannel]
        val byteBuffer: ByteBuffer = ByteBuffer.allocate(fileChannel.size().toInt)
        byteBuffer.order(ByteOrder.nativeOrder())
        fileChannel.read(byteBuffer)
        byteBuffer.flip()
        byteBuffer
      }


      def setTensorFloat(): Unit = {
        val data = loadData(file).asFloatBuffer()
        val array = new Array[Float](data.limit())
        data.get(array)
        tensor.asInstanceOf[Tensor[Float]].set(Storage(array), sizes = size)
      }

      def setTensorDouble(): Unit = {
        val data = loadData(file).asDoubleBuffer()
        val array = new Array[Double](data.limit())
        data.get(array)
        tensor.asInstanceOf[Tensor[Double]].set(Storage(array), sizes = size)
      }
    }

    tensor
  }

  def flattenModules[T: ClassTag](model: Module[T], modules: ArrayBuffer[TensorModule[T]])
                                 (implicit ev: TensorNumeric[T]): Unit = {
    model match {
      case container : Container[_, _, T] =>
        if (container.modules.nonEmpty) {
          for (i <- container.modules) {
            flattenModules(i.asInstanceOf[Module[T]], modules)
          }
        }
      case _ =>
        modules += model.asInstanceOf[TensorModule[T]]
    }
  }

  private[this] def formatName(name: String): String = {
    name.replaceAll("/", "_")
  }

  def compareAllLayers[T: ClassTag](modules: ArrayBuffer[TensorModule[T]], identity: String,
                                    direction: Direction)(implicit ev: TensorNumeric[T]): Unit = {
    def compare(tensor: Tensor[T], name: String, key: String): Unit = {
      val binaryName = key + formatName(name)
      val layerName = key + name
      val tensorFromBinary = loadTensor(binaryName, tensor.size(), identity)
      if (tensorFromBinary.nElement() > 0) {
        require(cumulativeError(tensor, tensorFromBinary, layerName) == 0, s"layer is different")
      }
    }

    for (i <- modules.indices) {
      val name = modules(i).getName()
      direction match {
        case Forward => compare(modules(i).output, name, Collect.leadOfForward)
        case Backward => compare(modules(i).gradInput, name, Collect.leadOfBackward)
        case _ => throw new RuntimeException(s"wrong direction")
      }
    }
  }

  def loadParameters[T: ClassTag](modules: ArrayBuffer[TensorModule[T]], identity: String)
                                 (implicit ev: TensorNumeric[T]): Unit = {
    for (i <- modules.indices) {
      val para = modules(i).parameters()
      if (para != null) {
        for (j <- para._1.indices) {
          val binName = Collect.leadOfForward + formatName(modules(i).getName()) + ".Wght." + j
          para._1(j).copy(loadTensor(binName, para._1(j).size(), identity))
        }
      }
    }
  }

  def compareParameters[T: ClassTag](modules: ArrayBuffer[TensorModule[T]], identity: String)
                                    (implicit ev: TensorNumeric[T]): Unit = {
    for (i <- modules.indices) {
      val para = modules(i).parameters()
      if (para != null) {
        for (j <- para._2.indices) {
          val binName = Collect.leadOfBackward + formatName(modules(i).getName()) + ".Grad." + j
          val tensor = loadTensor(binName, para._2(j).size(), identity)

          if (tensor.nElement() > 0) {
            cumulativeError(para._2(j), tensor, "GradWeight " + modules(i).getName())
          }
        }
      }
    }
  }

  def randTimes(): Int = util.Random.nextInt(5) + 1
}

/**
 * Call "collect" command, which is a method to collect output binary files.
 * It's similar to "caffe collect", the difference is that it supports collect
 * single layers output and gradient through make a fake gradOutput/top_diff.
 */
object Collect {
  val leadOfForward = "Fwrd_"
  val leadOfBackward = "Bwrd_"

  val tmpdir = System.getProperty("java.io.tmpdir")

  def hasCollect: Boolean = {
    val collectPath = System.getProperty("collect_location")
    val exitValue = if (collectPath != null) s"ls $collectPath".! else "which collect".!
    exitValue == 0
  }

  /**
   * save the prototxt to a temporary file and call collect
   * @param prototxt prototxt with string
   * @return the middle random number in temporary file, which is an identity for getTensor.
   */
  def run(prototxt: String, singleLayer: Boolean = true): String = {
    def saveToFile(prototxt: String, name: String): String = {
      val tmpFile = java.io.File.createTempFile(name, ".prototxt")
      val absolutePath = tmpFile.getAbsolutePath

      println(s"prototxt is saved to $absolutePath")

      val writer = new PrintWriter(tmpFile)
      writer.println(prototxt)
      writer.close()

      absolutePath
    }

    if (! hasCollect) {
      throw new RuntimeException(s"Can't find collect command. Have you copy to the PATH?")
    }

    val file = saveToFile(prototxt, "UnitTest.") // UnitTest ends with dot for getting random number
    val identity = file.split("""\.""").reverse(1) // get the random number

    val cmd = Seq("collect", "--model", file, "--type", "float", "--identity", identity)
    val exitValue = if (singleLayer) {
      Process(cmd :+ "--single", new File(tmpdir)).!
    } else {
      Process(cmd, new File(tmpdir)).!
    }

    require(exitValue == 0, s"Something wrong with collect command. Please check it.")

    identity
  }
}

/**
 * Fake Dropout layer, which is only return output in updateOutput
 * and gradOutput in updateGradInput.
 * Because the default Dropout will generate random output.
 * @param initP same as real Dropout
 * @param inPlace same as real Dropout
 * @param scale same as real Dropout
 * @param ev$1 same as real Dropout
 * @param ev same as real Dropout
 * @tparam T same as real Dropout
 */
class Dropout[@specialized(Float, Double) T: ClassTag](
    val initP: Double = 0.5,
    val inPlace: Boolean = false,
    var scale: Boolean = true)(implicit ev: TensorNumeric[T])
    extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    this.output.resizeAs(input).copy(input)
    input
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput.resizeAs(gradOutput).copy(gradOutput)
    this.gradInput
  }

  override def toString: String = {
    s"test.Dropout"
  }
}
