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

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}

import scala.reflect.ClassTag

object Tools {
  def Error[@specialized(Float, Double) T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T])(
      implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() == tensor2.nElement())
    var ret = 0.0
    for (i <- 0 until tensor1.nElement()) {
      ret += math.abs(ev.toType[Double](tensor1.storage().array()(i)) -
                      ev.toType[Double](tensor2.storage().array()(i)))
    }
    ret
  }

  def CumulativeError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)(
      implicit ev: TensorNumeric[T]): Double = {
    val ret = Error[T](tensor1, tensor2)
    println((msg, "CUMULATIVE ERROR:", ret).productIterator.mkString(" ").toUpperCase)
    ret
  }

  def AverageError[T: ClassTag](tensor1: Tensor[T], tensor2: Tensor[T], msg: String)(
      implicit ev: TensorNumeric[T]): Double = {
    require(tensor1.nElement() > 0)
    val ret = Error[T](tensor1, tensor2) / tensor1.nElement()
    println((msg, "AVERAGE ERROR:", ret).productIterator.mkString(" ").toUpperCase)
    ret
  }

  def AverageError[T: ClassTag](m1: Map[String, Tensor[T]],
                                m2: Map[String, Tensor[T]],
                                err: Map[String, Double])(implicit ev: TensorNumeric[T]): Unit = {
    require(m1.keySet == m2.keySet)
    require(m1.keySet subsetOf err.keySet)

    val maxLen = m1.keysIterator.reduceLeft((x, y) => if (x > y) x else y)

    m1.keySet.foreach(i => {
      val error = Error(m1(i), m2(i)) / m1(i).nElement()
      printf("%20s = %E\n", i.toUpperCase(), error)
    })
  }

  def AverageAll[T: ClassTag](tensor1 : Tensor[T],
                              msg : String = "Unknown")(implicit ev : TensorNumeric[T]): Unit = {
    val sum = tensor1.storage().array().foldLeft(ev.fromType[Int](0))((l,r) => ev.plus(l,r))
    val num = ev.fromType[Int](tensor1.nElement())
    println(("AVERGE", msg, ev.divide(sum, num)).productIterator.mkString(" ").toUpperCase())
  }

  def PrintTensor[T: ClassTag](tensor : Tensor[T],
                               num: Int = 16,
                               msg: String = "Unknown")(implicit ev: TensorNumeric[T]): Unit = {
    println(msg.toUpperCase)
    for (i <- 0 until(num)) {
      println((i, ev.toType[Double](tensor.storage().array()(i))).productIterator.mkString("\t"))
    }
  }

  def loadData(name : String) : ByteBuffer = {
    val fileChannel : FileChannel = Files.newByteChannel(Paths.get(name),
                                                         StandardOpenOption.READ).asInstanceOf[FileChannel]
    val byteBuffer : ByteBuffer = ByteBuffer.allocate(fileChannel.size().toInt)
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
  def GetTensorFloat(name : String, size : Array[Int]) : Tensor[Float] = {
    val tensor = Tensor[Float]()
    val data = Tools.loadData("/tmp/" + name + ".bin").asFloatBuffer()
    val array = new Array[Float](data.limit())
    data.get(array)
    tensor.set(Storage(array), sizes = size)

    tensor
  }

  def GetTensorDouble(name : String, size : Array[Int]) : Tensor[Double] = {
    val tensor = Tensor[Double]()
    val data = Tools.loadData("/tmp/" + name + ".bin").asDoubleBuffer()
    val array = new Array[Double](data.limit())
    data.get(array)
    tensor.set(Storage(array), sizes = size)

    tensor
  }

  def GetRandTimes(): Int = 10
}
