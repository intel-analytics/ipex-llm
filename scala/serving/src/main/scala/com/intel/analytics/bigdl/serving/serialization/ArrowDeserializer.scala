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

package com.intel.analytics.bigdl.serving.serialization

import java.io.ByteArrayInputStream
import java.util.Base64

import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.{Float4Vector, IntVector, VectorSchemaRoot}
import org.apache.arrow.vector.holders.NullableIntHolder
import org.apache.arrow.vector.ipc.ArrowStreamReader

/**
 * Arrow deserializer to deserialize arrow buffer for Scala/Java frond end
 * The frontend will get Array of data and Array of shape
 * and then make string based on them
 */
class ArrowDeserializer {
  /**
   * Get data and shape Array from VectorSchemaRoot
   * @param vectorSchemaRoot VectorSchemaRoot read from Arrow buffer
   * @return Tuple, (dataArray, shapeArray)
   */
  def getFromSchemaRoot(vectorSchemaRoot: VectorSchemaRoot): (Array[Float], Array[Int]) = {
    val dataVector = vectorSchemaRoot.getVector("data").asInstanceOf[Float4Vector]
    val shapeVector = vectorSchemaRoot.getVector("shape").asInstanceOf[IntVector]
    val dataArray = new Array[Float](dataVector.getValueCount)
    (0 until dataArray.size).foreach(i => dataArray(i) = dataVector.get(i))
    var shapeArray = Array[Int]()
    val nullableHolder = new NullableIntHolder()
    (0 until dataArray.size).foreach(i => {
      shapeVector.get(i, nullableHolder)
      if (nullableHolder.isSet == 1) {
        shapeArray = shapeArray :+ nullableHolder.value
      }
    })
    (dataArray, shapeArray)
  }

  /**
   * Deserialize base64 string to Array of (dataArray, shapeArray)
   * some models would have multiple output, aka. Table
   * Thus, Array is used to represent the multiple output
   * @param b64string base64 string to decode
   * @return
   */
  def create(b64string: String): Array[(Array[Float], Array[Int])] = {
    var result = Array[(Array[Float], Array[Int])]()
    val readAllocator = new RootAllocator(Int.MaxValue)
    val byteArr = Base64.getDecoder.decode(b64string)
    val reader = new ArrowStreamReader(new ByteArrayInputStream(byteArr), readAllocator)
    val schema = reader.getVectorSchemaRoot.getSchema
    while (reader.loadNextBatch()) {
      val vectorSchemaRoot = reader.getVectorSchemaRoot
      result = result :+ getFromSchemaRoot(vectorSchemaRoot)
    }
//    readAllocator.close()
    reader.close()
    result
  }
  def getJsonString(arr: Array[(Array[Float], Array[Int])]): String = {
    val strArr = arr.map(dataAndShape => {
      val dataStr = dataAndShape._1.mkString("[", ",", "]")
      val shapeStr = dataAndShape._2.mkString("[", ",", "]")
      s"""{"data":$dataStr,"shape":$shapeStr}"""
    })
    var str = ""
    strArr.foreach(s => str += s)
    str
  }
}

/**
 * Wrap class ArrowDeserializer to deserialize base64 string
 * to Array of (dataArray, shapeArray)
 */
object ArrowDeserializer {
  def apply(b64string: String): String = {
    val deserializer = new ArrowDeserializer()
    val arr = deserializer.create(b64string)
    deserializer.getJsonString(arr)
  }
  def getArray(b64string: String): Array[(Array[Float], Array[Int])] = {
    val deserializer = new ArrowDeserializer()
    val arr = deserializer.create(b64string)
    arr
  }
}
