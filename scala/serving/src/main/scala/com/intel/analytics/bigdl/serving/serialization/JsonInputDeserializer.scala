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

import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.module.SimpleModule
import com.fasterxml.jackson.databind.node.{ArrayNode, IntNode, ObjectNode, TextNode}
import com.fasterxml.jackson.databind.{DeserializationContext, JsonDeserializer, JsonNode, ObjectMapper}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.serving.preprocessing.PreProcessing

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class JsonInputDeserializer(preProcessing: PreProcessing = null)
  extends JsonDeserializer[Activity]{
  var intBuffer: ArrayBuffer[Int] = null
  var floatBuffer: ArrayBuffer[Float] = null
  var stringBuffer: ArrayBuffer[String] = null
  var shapeBuffer: ArrayBuffer[Int] = null
  var valueCount: Int = 0
  var shapeMask: Map[Int, Boolean] = null
  override def deserialize(p: JsonParser, ctxt: DeserializationContext): Activity = {
    val oc = p.getCodec
    val node = oc.readTree[JsonNode](p)
    val inputsIt = node.get("instances").get(0).elements()
    val tensorBuffer = new ArrayBuffer[Tensor[Float]]()
    while (inputsIt.hasNext) {
      initBuffer()
      parse(inputsIt.next(), 0)
      if (shapeBuffer.isEmpty) shapeBuffer.append(1)
      if (!floatBuffer.isEmpty) {
        tensorBuffer.append(Tensor[Float](floatBuffer.toArray, shapeBuffer.toArray))
      }
      else if (!stringBuffer.isEmpty) {
        if (preProcessing == null) {
          throw new Error("No PreProcessing provided!")
        }
        tensorBuffer.append(preProcessing.decodeImage(stringBuffer.head))
      }
      else {
        // add string, string tensor, sparse tensor in the future
        throw new Error("???")
      }

    }
    T.array(tensorBuffer.toArray)
  }
  def parse(node: JsonNode, currentShapeDim: Int): Unit = {
    if (node.isInstanceOf[ArrayNode]) {

      val iter = node.elements()
      if (shapeMask.get(currentShapeDim) == None) {
        shapeBuffer.append(node.size())
        shapeMask += (currentShapeDim -> true)
      }
      while (iter.hasNext) {
        parse(iter.next(), currentShapeDim + 1)
      }
    } else if (node.isInstanceOf[TextNode]) {
      stringBuffer.append(node.asText())
    } else if (node.isInstanceOf[ObjectNode]) {
      // currently used for SparseTensor only maybe
    } else {
      // v1: int, float, double would all parse to float
      floatBuffer.append(node.asDouble().toFloat)
    }

  }
  def initBuffer(): Unit = {
    floatBuffer = new ArrayBuffer[Float]()
    shapeBuffer = new ArrayBuffer[Int]()
    stringBuffer = new ArrayBuffer[String]()
    shapeMask = Map[Int, Boolean]()
  }

}
object JsonInputDeserializer {
  def deserialize(str: String, preProcessing: PreProcessing = null): Activity = {
    val mapper = new ObjectMapper()
    val module = new SimpleModule()
    module.addDeserializer(
      classOf[Activity], new JsonInputDeserializer(preProcessing))
    mapper.registerModule(module)
    mapper.readValue(str, classOf[Activity])
  }
}

