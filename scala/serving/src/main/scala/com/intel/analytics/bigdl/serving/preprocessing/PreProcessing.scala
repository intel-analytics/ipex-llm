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

package com.intel.analytics.bigdl.serving.preprocessing

import com.intel.analytics.bigdl.dllib.feature.image.OpenCVMethod
import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import org.opencv.imgcodecs.Imgcodecs
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer
import com.intel.analytics.bigdl.orca.inference.{EncryptSupportive, InferenceSupportive}
import com.intel.analytics.bigdl.serving.ClusterServing
import com.intel.analytics.bigdl.serving.http.Instances
import com.intel.analytics.bigdl.serving.pipeline.RedisUtils
import com.intel.analytics.bigdl.serving.serialization.{JsonInputDeserializer, StreamSerializer}
import com.intel.analytics.bigdl.serving.utils.{ClusterServingHelper, Conventions}
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

import scala.collection.JavaConverters._
import redis.clients.jedis.Jedis

class PreProcessing()
  extends EncryptSupportive with InferenceSupportive {
  val logger = Logger.getLogger(getClass)
  if (ClusterServing.helper == null) {
    ClusterServing.helper = new ClusterServingHelper
  }
  val helper = ClusterServing.helper

  var byteBuffer: Array[Byte] = null
  def getInputFromInstance(instance: Instances): Seq[Activity] = {
    instance.instances.flatMap(insMap => {
      val oneInsMap = insMap.map(kv =>
        if (kv._2.isInstanceOf[String]) {
          if (kv._2.asInstanceOf[String].contains("|")) {
            (kv._1, decodeString(kv._2.asInstanceOf[String]))
          }
          else {
            (kv._1, decodeImage(kv._2.asInstanceOf[String]))

          }
        }
        else {
          (kv._1, decodeTensor(kv._2.asInstanceOf[(
            ArrayBuffer[Int], ArrayBuffer[Float], ArrayBuffer[Int], ArrayBuffer[Int])]))
        }
      ).toList
      val arr = oneInsMap.map(x => x._2)
      Seq(T.array(arr.toArray))
    })
  }

  def decodeArrowBase64(key: String, s: String, serde: String = ""): Activity = {
    try {

      val instance = if (serde == "stream") {
        Seq(JsonInputDeserializer.deserialize(s, this))

      } else {
        byteBuffer = java.util.Base64.getDecoder.decode(s)
        val ins = Instances.fromArrow(byteBuffer)
        getInputFromInstance(ins)
      }

      val kvMap = instance
      kvMap.head
    } catch {
      case e: Exception =>
        logger.error(s"Preprocessing error, msg ${e.getMessage}")
        logger.error(s"Error stack trace ${e.getStackTrace.mkString("\n")}")
        val tmpJedis = RedisUtils.getRedisClient(ClusterServing.jedisPool)
        val hKey = Conventions.RESULT_PREFIX + ClusterServing.helper.jobName + ":" + key

        val hValue = Map[String, String]("value" -> "NaN").asJava
        tmpJedis.hset(hKey, hValue)
        tmpJedis.close()
        null
    }
  }
  def decodeString(s: String): Tensor[String] = {
    val eleList = s.split("\\|")
    val tensor = Tensor[String](eleList.length)
    (1 to eleList.length).foreach(i => {
      tensor.setValue(i, eleList(i - 1))
    })
    tensor
  }

  def decodeImage(s: String, idx: Int = 0): Tensor[Float] = {
    byteBuffer = if (helper.recordEncrypted) {
      val bytes = timing(s"base64decoding") {
        logger.debug("String size " + s.length)
        java.util.Base64.getDecoder.decode(s)
      }
      timing(s"decryption with gcm 128") {
        logger.debug("Byte size " + bytes.size)
        decryptBytesWithAESGCM(bytes,
          Conventions.RECORD_SECURED_SECRET, Conventions.RECORD_SECURED_SALT)
      }
    } else {
      java.util.Base64.getDecoder.decode(s)
    }
    val mat = OpenCVMethod.fromImageBytes(byteBuffer, Imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
    if (helper.imageResize != "") {
      val hw = helper.imageResize.split(",")
      require(hw.length == 2, "Image dim must be 2")
      Imgproc.resize(mat, mat, new Size(hw(0).trim.toInt, hw(1).trim.toInt))
    }
//    Imgproc.resize(mat, mat, new Size(224, 224))
    val (height, width, channel) = (mat.height(), mat.width(), mat.channels())

    val arrayBuffer = new Array[Float](height * width * channel)
    OpenCVMat.toFloatPixels(mat, arrayBuffer)

    val imageTensor = Tensor[Float](arrayBuffer, Array(height, width, channel))
    if (helper.chwFlag) {
      imageTensor.transpose(1, 3)
        .transpose(2, 3).contiguous()
    } else {
      imageTensor
    }
  }
  def decodeTensor(info: (ArrayBuffer[Int], ArrayBuffer[Float],
    ArrayBuffer[Int], ArrayBuffer[Int])): Tensor[Float] = {
    val data = info._2.toArray

    val shape = info._1.toArray
    if (info._3.size == 0) {
      Tensor[Float](data, shape)
    } else {
      val indiceData = info._4.toArray
      val indiceShape = info._3.toArray
      var indice = new Array[Array[Int]](0)
      val colLength = indiceShape(1)
      var arr: Array[Int] = null
      (0 until indiceData.length).foreach(i => {
        if (i % colLength == 0) {
          arr = new Array[Int](colLength)
        }
        arr(i % colLength) = indiceData(i)
        if ((i + 1) % colLength == 0) {
          indice = indice :+ arr
        }
      })
      Tensor.sparse(indice, data, shape)
    }

  }

}
