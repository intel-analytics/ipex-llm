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

/*
package com.intel.analytics.zoo.serving

import java.nio.file.{Files, Paths}
import java.util.{Base64, UUID}

import com.intel.analytics.zoo.serving.http._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable
import scala.util.Random

class FrontendDomainsSpec extends FlatSpec with Matchers with BeforeAndAfter with Supportive {

  val random = new Random()

  "ServingError" should "serialized as json" in {
    val message = "contentType not supported"
    val error = ServingError(message)
    error.toString should include(s""""error" : "$message"""")
  }

  "Feature" should "serialized and deserialized as json" in {
    val image1 = new ImageFeature("aW1hZ2UgYnl0ZXM=")
    val image2 = new ImageFeature("YXdlc29tZSBpbWFnZSBieXRlcw==")
    val image3Path = getClass().getClassLoader()
      .getResource("imagenet/n02110063/n02110063_15462.JPEG").getFile()
    val byteArray = Files.readAllBytes(Paths.get(image3Path))
    val image3 = new ImageFeature(Base64.getEncoder().encodeToString(byteArray))
    val instance3 = mutable.LinkedHashMap[String, Any]("image" -> image3, "caption" -> "dog")
    val inputs = Instances(List.range(0, 2).map(i => instance3))
    val json = timing("serialize")() {
      JsonUtil.toJson(inputs)
    }
    // println(json)
    val obj = timing("deserialize")() {
      JsonUtil.fromJson(classOf[Instances], json)
    }
    obj.instances.size should be(2)
  }

  "BytesPredictionInput" should "works well" in {
    val bytesStr = "aW1hZ2UgYnl0ZXM="
    val input = BytesPredictionInput(bytesStr)
    input.toHash().get("data") should equal(bytesStr)
  }

  "PredictionOutput" should "works well" in {
    val uuid = UUID.randomUUID().toString
    val result = "mock-result"
    val out = PredictionOutput(uuid, result)
    out.uuid should be(uuid)
    out.result should be(result)
  }

  val instancesJson =
    """{
      |"instances": [
      |   {
      |     "tag": "foo",
      |     "signal": [1, 2, 3, 4, 5],
      |     "sensor": [[1, 2], [3, 4]]
      |   },
      |   {
      |     "tag": "bar",
      |     "signal": [3, 4, 1, 2, 5],
      |     "sensor": [[4, 5], [6, 8]]
      |   }
      |]
      |}
      |""".stripMargin
  "Instances" should "works well" in {
    val instances = JsonUtil.fromJson(classOf[Instances], instancesJson)
    instances.instances.size should be(2)

    val intScalar = 12345
    val floatScalar = 3.14159
    val stringScalar = "hello, world. hello, arrow."
    val intTensor = List.range(0, 1000).map(i => random.nextInt(10000))
    val floatTensor = List.range(0, 1000).map(i => random.nextFloat())
    val stringTensor = List("come", "on", "united")
    val intTensor2 = List(List(1, 2), List(3, 4), List(5, 6))
    val floatTensor2 =
      List(
        List(
          List(.2f, .3f),
          List(.5f, .6f)),
        List(
          List(.2f, .3f),
          List(.5f, .6f)))
    val stringTensor2 =
      List(
        List(
          List(
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united")),
          List(
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"))
        ),
        List(
          List(
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united")),
          List(
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"),
            List("come", "on", "united"))
        )
      )
    val instance = mutable.LinkedHashMap(
      "intScalar" -> intScalar,
      "floatScalar" -> floatScalar,
      "stringScalar" -> stringScalar,
      "intTensor" -> intTensor,
      "floatTensor" -> floatTensor,
      "stringTensor" -> stringTensor,
      "intTensor2" -> intTensor2,
      "floatTensor2" -> floatTensor2,
      "stringTensor2" -> stringTensor2
    )

    val instances2 = Instances(instance, instance)

    val json2 = timing("json serialization")() {
      JsonUtil.toJson(instances2)
    }
    val instances3 = timing("json deserialization")() {
      JsonUtil.fromJson(classOf[Instances], json2)
    }
    // println("json: " + json2)
    // println("json serialized size: " + json2.getBytes.length)

    val tensors = instances3.constructTensors()
    val schemas = instances3.makeSchema(tensors)

    val (shape1, data1) = Instances.transferListToTensor(intTensor)
    shape1.reduce(_ * _) should be(data1.size)
    val (shape2, data2) = Instances.transferListToTensor(intTensor2)
    shape2.reduce(_ * _) should be(data2.size)
    val (shape3, data3) = Instances.transferListToTensor(floatTensor2)
    shape3.reduce(_ * _) should be(data3.size)
    val (shape4, data4) = Instances.transferListToTensor(stringTensor2)
    shape4.reduce(_ * _) should be(data4.size)

    val arrowBytes = timing("arrow serialization")() {
      instances3.toArrow()
    }
    // println("arrow:" + new String(arrowBytes))
    // println("arrow serialized size: ", arrowBytes.length)
    val instances4 = timing("arrow deserialization")() {
      Instances.fromArrow(arrowBytes)
    }
    instances4.instances(0).get("intScalar") should be(Some(12345))
    instances4.instances(0).get("floatScalar") should be(Some(3.14159f))
    instances4.instances(0).get("stringScalar") should be(Some("hello, world. hello, arrow."))
    println(instances4.instances(0).get("intTensor"))
    println(instances4.instances(0).get("floatTensor"))
    println(instances4.instances(0).get("stringTensor"))
    println(instances4.instances(0).get("intTensor2"))
    println(instances4.instances(0).get("floatTensor2"))
    println(instances4.instances(0).get("stringTensor2"))
  }

  "Instances" should "works well too" in {
    List.range(0, 10).foreach(i => {
      val image3Path = getClass().getClassLoader()
        .getResource("imagenet/n02110063/n02110063_15462.JPEG").getFile()
      val byteArray = Files.readAllBytes(Paths.get(image3Path))
      val b64 = Base64.getEncoder().encodeToString(byteArray)
      val instance = mutable.LinkedHashMap("image" -> b64)
        .asInstanceOf[mutable.LinkedHashMap[String, Any]]
      val instances = Instances(List.range(0, 1).map(i => instance))

      val json = timing("json serialization")() {
        JsonUtil.toJson(instances)
      }
      val instances2 = timing("json deserialization")() {
        JsonUtil.fromJson(classOf[Instances], json)
        json
      }
      // println("json: " + json)
      println("json serialized size: " + json.getBytes.length)

      val arrowBytes = timing("arrow serialization")() {
        instances.toArrow()
      }
      val instances3 = timing("arrow deserialization")() {
        Instances.fromArrow(arrowBytes)
      }
      // println("arrow: " + new String(arrowBytes))
      println("arrow serialized size: " + arrowBytes.length)

      val data = List.range(0, 224).map(i => random.nextFloat())
      val data2 = List.range(0, 224).map(i => data)
      val data3 = List.range(0, 3).map(data2)
      val instance2 = mutable.LinkedHashMap(
        "feature" -> data3
      ).asInstanceOf[mutable.LinkedHashMap[String, Any]]
      val instances4 = Instances(List.range(0, 1).map(i => instance2))
      val json2 = timing("json serialization")() {
        JsonUtil.toJson(instances4)
      }
      val instances5 = timing("json deserialization")() {
        JsonUtil.fromJson(classOf[Instances], json2)
      }
      // println("json: " + json2)
      println("json serialized size: " + json2.getBytes.length)
      val arrowBytes2 = timing("arrow serialization")() {
        instances4.toArrow()
      }
      val instances6 = timing("arrow deserialization")() {
        Instances.fromArrow(arrowBytes2)
      }
      // println("arrow: " + new String(arrowBytes2))
      // println("arrow serialized size: " + arrowBytes2.length)

      val tensorFloat = List(
        List(1, 2),
        List(3, 4)
      )
      val instanceExample = mutable.LinkedHashMap("tensor" -> tensorFloat)
        .asInstanceOf[mutable.LinkedHashMap[String, Any]]
      val instancesExample = Instances(instanceExample)
      val arrowBytesExample = timing("arrow serialization")() {
        instancesExample.toArrow()
      }
      val b64Example = Base64.getEncoder().encodeToString(arrowBytesExample)
      // println("XXXXXXXXXXXXXXXXX:\n" + new String(arrowBytesExample))
      // println("arrow:\n " + b64Example)

      val arrowBytesPath = getClass().getClassLoader()
        .getResource("serving/arrowBytes").getFile()
      val b64f = scala.io.Source.fromFile(arrowBytesPath).mkString
      val bytes = java.util.Base64.getDecoder.decode(b64f)
      // println(new String(bytes))
      val instancesEx = timing("arrow deserialization")() {
        Instances.fromArrow(bytes)
      }
      instancesEx.instances.size should be (1)
      instancesEx.instances(0).get("my-img").size should be (1)
      // println(instancesEx)
    })
  }

  "sparse tensor" should "work" in {
    val shape = List(100, 10000, 10)
    val values = List(0.2f, 0.5f, 3.45f, 6.78f)
    val indices = List(List(1, 1, 1), List(2, 2, 2), List(3, 3, 3), List(4, 4, 4))
    val sparseTensor = SparseTensor(shape, values, indices)
    val intTensor2 = List(List(1, 2), List(3, 4), List(5, 6))
    val instance = mutable.LinkedHashMap(
      "sparseTensor" -> sparseTensor,
      "intTensor2" -> intTensor2
    ).asInstanceOf[mutable.LinkedHashMap[String, Any]]
    val instances = Instances(instance, instance)

    val json = timing("json serialization")() {
      JsonUtil.toJson(instances)
    }
    // println(json)
    val instances2 = timing("json deserialization")() {
      JsonUtil.fromJson(classOf[Instances], json)
    }
    // println(instances2)
    // println("json serialized size: " + json.getBytes.length)

    val arrowBytes = timing("arrow serialization")() {
      instances.toArrow()
    }
    val instances3 = timing("arrow deserialization")() {
      Instances.fromArrow(arrowBytes)
    }
    // println(instances3)
    // println("arrow serialized size: " + arrowBytes.length)
  }

}
*/
