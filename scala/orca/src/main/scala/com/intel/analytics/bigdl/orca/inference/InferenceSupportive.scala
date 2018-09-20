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

package com.intel.analytics.zoo.pipeline.inference

import com.intel.analytics.bigdl.dataset.Sample
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.tensor.Tensor
import java.util.{List => JList}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}
import java.util

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable.ArrayBuffer


trait InferenceSupportive {

  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    InferenceSupportive.logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  @inline
  private def product(input: JList[JInt]): Int = {
    var i = 0
    val length = input.size()
    var product = 1
    while (i < length) {
      product = product * input.get(i)
      i += 1
    }
    product
  }

  @inline
  private def toFloatArray(data: JList[JFloat]): Array[Float] = {
    val length = data.size()
    val result = new Array[Float](length)
    var i = 0
    while (i < length) {
      result(i) = data.get(i)
      i += 1
    }
    result
  }

  @inline
  private def toIntArray(data: JList[JInt]): Array[Int] = {
    val length = data.size()
    val result = new Array[Int](length)
    var i = 0
    while (i < length) {
      result(i) = data.get(i)
      i += 1
    }
    result
  }

  def transferBatchTensorToJListOfJListOfJTensor(batchTensor: Tensor[Float], batchSize: Int)
  : JList[JList[JTensor]] = {
    val batchTensorSize: Array[Int] = batchTensor.size()
    val batchShape: Array[Int] = batchTensorSize
    val outputLength = batchTensor.nElement() / batchSize
    val outputShape = batchShape.tail
    require(batchSize == batchShape.head, "batchSize should be the same, " +
      "please check if the batchSize is changed by the model")
    require(outputLength == outputShape.reduce(_ * _),
      "data length should be equal to the product of shape")
    val outputs = new util.ArrayList[JList[JTensor]]()
    var i = 0
    while(i < batchSize) {
      val storageOffset = batchTensor.storageOffset - 1 + i * outputLength
      val res = new Array[Float](outputLength)
      System.arraycopy(batchTensor.storage().array(), storageOffset, res, 0, res.length)
      val outputTensor = new JTensor(res, outputShape, false)
      outputs.add(util.Arrays.asList({outputTensor}))
      i += 1
    }
    outputs
  }

  def transferBatchTableToJListOfJListOfJTensor(batchTable: Table, batchSize: Int)
  : JList[JList[JTensor]] = {
    val tableBatches: Seq[JList[JList[JTensor]]] = batchTable.toSeq[Tensor[Float]].map(t =>
      transferBatchTensorToJListOfJListOfJTensor(t, batchSize)
    )
    var i = 0
    val outputs = new util.ArrayList[JList[JTensor]]()
    while(i < batchSize) {
      var j = 0
      val tensorList = new util.ArrayList[JTensor]()
      while(j < tableBatches.size) {
        val tableBatch_j = tableBatches(j)
        val tensorij = tableBatch_j.get(i).get(0)
        tensorList.add(tensorij)
        j += 1
      }
      outputs.add(tensorList)
      i += 1
    }
    outputs
  }

  def transferTensorToJTensor(input: Tensor[Float]): JTensor = {
    val storageOffset = input.storageOffset - 1
    val res = new Array[Float](input.nElement())
    System.arraycopy(input.storage().array(), storageOffset, res, 0, res.length)
    val outputShape = input.size()
    new JTensor(res, outputShape, false)
  }

  def transferListOfActivityToActivityOfBatch(inputs: JList[JList[JTensor]], batchSize: Int)
  : Activity = {
    require(batchSize == inputs.size, "batchSize should be the same")
    val head = inputs.get(0)
    val headLength = head.size()
    headLength match {
      case 0 => throw new InferenceRuntimeException("input of JList[JTensor] cannot be 0 length")
      case 1 =>
        var i = 0
        val tensors = ArrayBuffer[JTensor]()
        while (i < batchSize) {
          tensors += inputs.get(i).get(0)
          i += 1
        }
        transferTensorsToTensorOfBatch(tensors.toArray)
      case x =>
        val inputTable = T()
        val tensorsArray = (1 to x).map(i => ArrayBuffer[JTensor]())
        var i = 0
        while (i < batchSize) {
          val inputList = inputs.get(i)
          var j = 0
          while (j < x) {
            val tensors = tensorsArray(j)
            tensors += inputList.get(j)
            j += 1
          }
          i += 1
        }
        var j = 0
        while (j < x) {
          val tensors = tensorsArray(j)
          val batchTensor = transferTensorsToTensorOfBatch(tensors.toArray)
          inputTable.insert(batchTensor)
          j += 1
        }
        inputTable
    }
  }

  def transferTensorsToTensorOfBatch(tensors: Array[JTensor]): Tensor[Float] = {
    val batchSize = tensors.length
    val head = tensors.head
    val shape = (ArrayBuffer(batchSize) ++= head.getShape)
    val data = ArrayBuffer[Float]()
    var i = 0
    while (i < batchSize) {
      val tensor = tensors(i)
      val tensorData = tensor.getData
      data ++= tensorData
      i += 1
    }
    require(data.length == shape.reduce(_ * _),
      "data length should be equal to the product of shape")
    Tensor[Float](data.toArray, shape.toArray)
  }

}

object InferenceSupportive {
  val logger = LoggerFactory.getLogger(getClass)
}
