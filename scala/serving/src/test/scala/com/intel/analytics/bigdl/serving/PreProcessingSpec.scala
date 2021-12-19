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

package com.intel.analytics.bigdl.serving

import com.intel.analytics.bigdl.serving.http.{Instances, JsonUtil}
import com.intel.analytics.bigdl.serving.preprocessing.PreProcessing
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

class PreProcessingSpec extends FlatSpec with Matchers {
  "base64 string to tensor" should "work" in {
  }
  "create buffer" should "work" in {
    val a = Array(3, 224, 224)
  }
  "generate table" should "work" in {
    val t = List((1, 2))
    def d(tuples: (Any, Any)*): Unit = {
      tuples
    }
    d(t: _*)
  }
  "table copy" should "work" in {
    val arr = Array(1, 2, 3)
    def cp(t1: Int, t2: Int, t3: Int, t4: Int): Unit = {
      None
    }
  }
  "decode tensor" should "work" in {
    val iData = ArrayBuffer(1, 2, 3, 1, 2, 3)
    val iShape = ArrayBuffer(2, 3)
    val data = ArrayBuffer(3f, 4, 5)
    val shape = ArrayBuffer(100, 10000)
    val pre = new PreProcessing()
    val info = (shape, data, iShape, iData)
    val a = pre.decodeTensor(info)
    a
  }
  "decode string tensor" should "work" in {
    val pre = new PreProcessing()
    val str = "abc|dff|aoa"
    val tensor = pre.decodeString(str)
    assert(tensor.valueAt(1) == "abc")
    assert(tensor.valueAt(2) == "dff")
    assert(tensor.valueAt(3) == "aoa")
  }
  "parse json to tensor" should "work" in {
    val instancesJson =
      """{
        |"instances": [
        |   {
        |     "tag": "foo",
        |     "signal": [1, 2, 3, 4, 5],
        |     "sensor": [[1, 2], [3, 4]]
        |   }
        |]
        |}
        |""".stripMargin

    val instances = JsonUtil.fromJson(classOf[Instances], instancesJson)
    val arrowBytes = instances.toArrow()
    val arrowInstance = Instances.fromArrow(arrowBytes)
    val pre = new PreProcessing()
    val t = pre.getInputFromInstance(arrowInstance)
    assert(t.head.toTable.keySet.size == 3)

  }
}
