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

package com.intel.analytics.sparkdl.nn.mkl

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.util.NoSuchElementException

import com.intel.analytics.sparkdl.nn.{Module, TensorModule}
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object Tools {
  def error[@specialized(Float, Double) T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T])(
      implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() == tensor2.nElement())
    var ret = 0.0
    for (i <- 0 until tensor1.nElement()) {
      ret += math.abs(
        ev.toType[Double](tensor1.storage().array()(i)) -
          ev.toType[Double](tensor2.storage().array()(i)))
    }
    ret
  }

  def cumulativeError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)(
      implicit ev: TensorNumeric[T]): Double = {
    val ret = error[T](tensor1, tensor2)
    println((msg, "CUMULATIVE ERROR:", ret).productIterator.mkString(" ").toUpperCase)
    ret
  }

  def averageError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)(
      implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() > 0)
    val ret = error[T](tensor1, tensor2) / tensor1.nElement()
    println((msg, "AVERAGE ERROR:", ret).productIterator.mkString(" ").toUpperCase)
    ret
  }

  def averageError[T: ClassTag](m1: Map[String, Tensor[T]],
                                m2: Map[String, Tensor[T]],
                                err: Map[String, Double])(implicit ev: TensorNumeric[T]): Unit = {
    require(m1.keySet == m2.keySet)
    require(m1.keySet subsetOf err.keySet)

    val maxLen = m1.keysIterator.reduceLeft((x, y) => if (x > y) x else y)

    m1.keySet.foreach(i => {
      val err = error(m1(i), m2(i)) / m1(i).nElement()
      printf("%20s = %E\n", i.toUpperCase(), err)
    })
  }

  def averageAllTensors[T: ClassTag](tensor1: Tensor[T], msg: String = "Unknown")(
      implicit ev: TensorNumeric[T]): Unit = {
    val sum = tensor1.storage().array().foldLeft(ev.fromType[Int](0))((l, r) => ev.plus(l, r))
    val num = ev.fromType[Int](tensor1.nElement())
    println(("AVERGE", msg, ev.divide(sum, num)).productIterator.mkString(" ").toUpperCase())
  }

  def printTensor[T: ClassTag](tensor: Tensor[T], num: Int = 16, msg: String = "Unknown")(
      implicit ev: TensorNumeric[T]): Unit = {
    println(msg.toUpperCase)
    for (i <- 0 until (num)) {
      println((i, ev.toType[Double](tensor.storage().array()(i))).productIterator.mkString("\t"))
    }
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

  // TODO the two methods below (GetTensorFloat & GetTensorDouble) should be re-implemented.

  /*
   * @brief read "/tmp/<name>.bin" file to Tensor, which is used for comparing
   *        with IntelCaffe with MKL-DNN
   */
  def getTensor[T : ClassTag](name: String, size: Array[Int],
                              suffix : String = "")(implicit ev : TensorNumeric[T]): Tensor[T] = {
    val tensor = Tensor[T]()
    val prefix = "/tmp/" + name + ".bin"
    val file = prefix + (if (!suffix.isEmpty) { "." + suffix } else "")

    if (Files.exists(Paths.get(file))) {
      tensor match {
        case _:Tensor[Float] => setTensorFloat()
        case _:Tensor[Double] => setTensorDouble()
      }

      def setTensorFloat(): Unit = {
        val data = Tools.loadData(file).asFloatBuffer()
        val array = new Array[Float](data.limit())
        data.get(array)
        tensor.asInstanceOf[Tensor[Float]].set(Storage(array), sizes = size)
      }

      def setTensorDouble(): Unit = {
        val data = Tools.loadData(file).asDoubleBuffer()
        val array = new Array[Double](data.limit())
        data.get(array)
        array.asInstanceOf[Array[T]]
        tensor.asInstanceOf[Tensor[Double]].set(Storage(array), sizes = size)
      }
    }

    tensor
  }

  // TODO delete this method.
  def getTensorFloat(name: String, size: Array[Int],
                     suffix : String = ""): Tensor[Float] = {
    val tensor = Tensor[Float]()
    val file = if (!suffix.isEmpty) {
      "/tmp/" + name + ".bin." + suffix
    } else {
      "/tmp/" + name + ".bin"
    }
    val data = Tools.loadData(file).asFloatBuffer()
    val array = new Array[Float](data.limit())
    data.get(array)
    tensor.set(Storage(array), sizes = size)

    tensor
  }

  def getPidFromString(log : String) : String = {
    val pattern = "SUFFIX WITH PID IS ([0-9]+)\n".r
    (pattern.findFirstIn(log)) match {
      case Some(pattern(v)) => v
      case None    => throw new NoSuchElementException(s"dont found in ${log}")
    }
  }

  def flattenModules(model: Module[Tensor[Float], Tensor[Float], Float],
                     modules: ArrayBuffer[TensorModule[Float]]) : Unit = {
    if (model.modules.length >= 1) {
      for (i <- model.modules) {
        flattenModules(i.asInstanceOf[TensorModule[Float]], modules)
      }
    } else {
      modules += model.asInstanceOf[TensorModule[Float]]
    }
  }

  def getRandTimes(): Int = 3

  def getCaffeHome() : String = "/home/wyz/workspace/caffe.intel/"
  def getCollectCmd() : String = getCaffeHome() + "build/tools/caffe collect --model"
  def getModuleHome() : String = "/home/wyz/workspace/performance/models_perf/models/"
}

// Just for test, get rid of random.
class Dropout[@specialized(Float, Double) T: ClassTag]
( val initP: Double = 0.5,
  val inplace: Boolean = false,
  var scale: Boolean = true)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    this.output.resizeAs(input).copy(input)
    input
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput.resizeAs(gradOutput).copy(gradOutput)
    this.gradInput
  }

  override def toString(): String = {
    s"test.Dropout"
  }
}

/*
 * For truncate the float or double
 */
class Dummy[@specialized(Float, Double) T: ClassTag]
(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput = gradOutput.apply1(
      x => ev.fromType[Double]((math floor ev.toType[Double](x)  * 1e5) / 1e5)
    )

    gradInput
  }
}

