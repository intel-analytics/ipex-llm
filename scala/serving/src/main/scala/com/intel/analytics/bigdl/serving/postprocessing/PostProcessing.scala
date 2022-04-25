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

package com.intel.analytics.bigdl.serving.postprocessing

import java.util.Base64

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.serving.serialization.ArrowSerializer
import com.intel.analytics.bigdl.serving.utils.TensorUtils


/**
 * PostProssing
 * PostProcessing contains two steps
 * step 1 is filter, which is optional,
 * used to transform output tensor to type wanted
 * step 2 is to ndarray string, which is mandatory
 * to parse tensor into readable string
 * this string could be parsed by json in Python to a list
 * @param tensor
 */
class PostProcessing(tensor: Tensor[Float], filter: String = "") {
  var t: Tensor[Float] = tensor
  val totalSize = TensorUtils.getTotalSize(t)

  /**
   * Transform tensor into readable string,
   * could apply to any shape of tensor
   * @return
   */
  def tensorToNdArrayString(): String = {
    val sizeArray = t.size()
    var strideArray = Array[Int]()
    (0 until sizeArray.length).foreach(i => {
      var res: Int = 1
      (0 to i).foreach(j => {
        res *= sizeArray(sizeArray.length - 1 - j)
      })
      strideArray = strideArray :+ res
    })
    val flatTensor = t.resize(totalSize).toArray()
    var str: String = ""
    (0 until flatTensor.length).foreach(i => {
      (0 until sizeArray.length).foreach(j => {
        if (i % strideArray(j) == 0) {
          str += "["
        }
      })
      str += flatTensor(i).toString
      (0 until sizeArray.length).foreach(j => {
        if ((i + 1) % strideArray(j) == 0) {
          str += "]"
        }
      })
      if (i != flatTensor.length - 1) {
        str += ","
      }
    })
    str
  }
  /**
   * TopN filter, take 1-D size (n) tensor as input
   * @param topN
   * @return string, representing 2-D size (topN, 2) tensor
   */
  def rankTopN(topN: Int): String = {
    val list = TensorUtils.getTopN(topN, t)
    var res: String = ""
    res += "["
    (0 until list.size).foreach(i =>
      res += "[" + list(i)._1.toString + "," + list(i)._2.toString + "]"
    )
    res += "]"
    res
  }

  /**
   * Pick TopN value of output tensor
   * only (1) * record_size * box_value_number is supported
   * thus only 2 or 3 dimension is valid for now
   */
  def pickTopN(topN: Int): String = {
    require(t.dim() == 2 || t.dim() == 3,
      "pickTopN post-processing only take 2 or 3 output dim tensor")
    val thisT = if (t.dim() == 3) {
      t.squeeze(1)
    } else {
      t
    }
    require(thisT.dim() == 2,
      "Your input dim is 3 but squeeze operation fails, please open issue to BigDL team")
    var res: String = ""
    res += "["
    (1 to topN).foreach(topIdx => {
      res += "["
      (1 to t.size(2)).foreach(boxIdx => {
        res += t.valueAt(topIdx, boxIdx)
        if (boxIdx != thisT.size(2)) {
          res += ","
        }
      })
      res += "]"
    })
    res += "]"
    res
  }
  def processTensor(): String = {
    if (filter != "") {
      require(filter.last == ')',
        "please check your filter format, should be filter_name(filter_args)")
      require(filter.split("\\(").length == 2,
        "please check your filter format, should be filter_name(filter_args)")

      val filterType = filter.split("\\(").head
      val filterArgs = filter.split("\\(").last.dropRight(1).split(",")
      val res = filterType match {
        case "topN" =>
          require(filterArgs.length == 1, "topN filter only support 1 argument, please check.")
          rankTopN(filterArgs(0).toInt)
        case "pickTopN" =>
          require(filterArgs.length == 1, "pickTopN filter only support 1 argument, please check.")
          pickTopN(filterArgs(0).toInt)
        case _ => ""
      }
      res
    }
    else {
      tensorToNdArrayString()
    }
  }
}
object PostProcessing {
  /**
   *
   * @param t the result of prediction
   * @param filter custom postprocessing
   * @param index index of tensor to select, -1 means return tensor directly without select
   * @return
   */
  def apply(t: Activity, filter: String = "", index: Int = -1): String = {
    if (filter == "") {
      val byteArr = ArrowSerializer.activityBatchToByte(t, index)
      Base64.getEncoder.encodeToString(byteArr)
    }
    else {
      if (t.isTable) {
        var value = ""
        t.toTable.keySet.foreach(key => {
          val cls = new PostProcessing(t.toTable(key)
            .asInstanceOf[Tensor[Float]].select(1, index), filter)
          value += cls.processTensor()
        })
        value
      } else if (t.isTensor) {
        val cls = new PostProcessing(t.toTensor[Float].select(1, index), filter)
        cls.processTensor()
      } else {
        throw new Error("Your input for Post-processing is invalid, " +
          "neither Table nor Tensor, please check.")
      }
    }

  }
}
