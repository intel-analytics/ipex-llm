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

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.serving.postprocessing.PostProcessing
import com.intel.analytics.bigdl.serving.preprocessing.PreProcessing
import org.apache.logging.log4j.LogManager

/**
 *
 * @param modelKey Whether multiple Cluster Serving jobs share a same process (JVM)
 *                 If not sharing, modelKey should be null
 */
class ClusterServingInference(modelKey: String = null) {
  val logger = LogManager.getLogger(getClass)
  val helper = ClusterServing.helper
  val preProcessing = new PreProcessing()

  def singleThreadPipeline(in: List[(String, String, String)]): List[(String, String)] = {
    singleThreadInference(preProcess(in))
  }
  def multiThreadPipeline(in: List[(String, String, String)]): List[(String, String)] = {
    multiThreadInference(preProcess(in, true))
  }

  def preProcess(in: List[(String, String, String)],
                 multiThread: Boolean = false): List[(String, Activity)] = {
    val preProcessed = if (!multiThread) {
      in.map(item => {
        val uri = item._1
        val input = preProcessing.decodeArrowBase64(uri, item._2, item._3)
        (uri, input)
      })
    } else {
      val size = in.size
      in.grouped(size).flatMap(itemBatch => {
        (0 until size).toParArray.map(i => {
          val uri = itemBatch(i)._1
          val input = preProcessing.decodeArrowBase64(uri, itemBatch(i)._2, itemBatch(i)._3)
          (uri, input)
        })
      }).toList
    }
    preProcessed.filter(x => x._2 != null)
  }
  def singleThreadInference(in: List[(String, Activity)]): List[(String, String)] = {

    val postProcessed = in.map(pathByte => {
      try {
        val t = typeCheck(pathByte._2)
        val result = if (modelKey != null) {
          ClusterServing.jobModelMap(modelKey).doPredict(t)
        } else ClusterServing.model.doPredict(t)
        val value = PostProcessing(result.toTensor[Float], helper.postProcessing, -1)
        (pathByte._1, value)
      } catch {
        case e: Exception =>
          logger.error(s"${e.printStackTrace()}, " +
            s"Your input ${pathByte._1} format is invalid to your model, this record is skipped")
          (pathByte._1, "NaN")
      }
    })
    postProcessed
  }

  /** Deprecated
   * Current used for OpenVINO model, use multiple thread to inference, and single thread
   * to do other operations, normally only one model is used, and every thread in pipeline
   * try to get this model if it goes to inference stage
   * Do not need to set resize label, because only OpenVINO use it, and OpenVINO only support
   * fixed size of input, thus mutable batch size is not supported
   */
  def singleThreadBatchInference(in: List[(String, Activity)]): List[(String, String)] = {

    val postProcessed = in.grouped(helper.threadPerModel).flatMap(pathByte => {
      try {
        val thisBatchSize = pathByte.size
        val t = batchInput(pathByte, helper.threadPerModel,
          useMultiThreading = false, resizeFlag = false)
        dimCheck(t, "add", helper.modelType)
        val result =
          ClusterServing.model.doPredict(t)
        dimCheck(result, "remove", helper.modelType)
        dimCheck(t, "remove", helper.modelType)
        val kvResult =
          (0 until thisBatchSize).map(i => {
            val value = PostProcessing(result, helper.postProcessing, i + 1)
            (pathByte(i)._1, value)
          })
        kvResult
      } catch {
        case e: Exception =>
          logger.error(s"${e.printStackTrace()}, " +
            s"Your input format is invalid to your model, this batch is skipped")
          pathByte.map(x => (x._1, "NaN"))
      }
    })
    postProcessed.toList
  }

  def multiThreadInference(in: List[(String, Activity)]): List[(String, String)] = {

    val postProcessed = in.grouped(helper.threadPerModel).flatMap(itemBatch => {
      try {
        val size = itemBatch.size

        val t =
          batchInput(itemBatch, helper.threadPerModel, true, helper.resize)

        /**
         * addSingletonDimension method will modify the
         * original Tensor, thus if reuse of Tensor is needed,
         * have to squeeze it back.
         */
        val result = if (modelKey != null) {
          ClusterServing.jobModelMap(ClusterServing.helper.modelPath).doPredict(t)
        } else ClusterServing.model.doPredict(t)
        val kvResult =
          (0 until size).toParArray.map(i => {
            val value = PostProcessing(result, helper.postProcessing, i + 1)
            (itemBatch(i)._1, value)
          })
        kvResult
      } catch {
        case e: Exception =>
          logger.error(s"${e.printStackTrace()}, " +
            s"Your input format is invalid to your model, this batch is skipped")
          itemBatch.toParArray.map(x => (x._1, "NaN"))
      }
    })
    postProcessed.toList
  }

  def batchInput(seq: Seq[(String, Activity)],
                 batchSize: Int,
                 useMultiThreading: Boolean,
                 resizeFlag: Boolean = true): Activity = {
    val thisBatchSize = seq.size

    val inputSample = seq.head._2.toTable
    val kvTuples = inputSample.keySet.map(key => {
      (key, Tensor[Float](batchSize +:
        inputSample(key).asInstanceOf[Tensor[Float]].size()))
    }).toList
    val t = T(kvTuples.head, kvTuples.tail: _*)
    // Batch tensor and copy
    if (!useMultiThreading) {
      (0 until thisBatchSize).foreach(i => {
        val dataTable = seq(i)._2.toTable
        t.keySet.foreach(key => {
          t(key).asInstanceOf[Tensor[Float]].select(1, i + 1)
            .copy(dataTable(key).asInstanceOf[Tensor[Float]])
        })
      })
    } else {
      (0 until thisBatchSize).toParArray.foreach(i => {
        val dataTable = seq(i)._2.toTable
        t.keySet.foreach(key => {
          t(key).asInstanceOf[Tensor[Float]].select(1, i + 1)
            .copy(dataTable(key).asInstanceOf[Tensor[Float]])
        })
      })
    }
    // Resize and specific control
    if (resizeFlag) {
      t.keySet.foreach(key => {
        val singleTensorSize = inputSample(key).asInstanceOf[Tensor[Float]].size()
        var newSize = Array(thisBatchSize)
        for (elem <- singleTensorSize) {
          newSize = newSize :+ elem
        }
        t(key).asInstanceOf[Tensor[Float]].resize(newSize)
      })
    }
    if (t.keySet.size == 1) {
      t.keySet.foreach(key => {
        return t(key).asInstanceOf[Tensor[Float]]
      })
    }
    t
  }

  /**
   * Add or remove the singleton dimension for some specific model types
   * @param input the input to change dimension
   * @param op String, "add" or "remove"
   * @param modelType model type
   * @return input with dimension changed
   */
  @deprecated
  def dimCheck(input: Activity, op: String, modelType: String): Activity = {
    if (modelType == "openvino") {
      if (input.isTensor) {
        op match {
          case "add" => input.asInstanceOf[Tensor[Float]].addSingletonDimension()
          case _ => input.asInstanceOf[Tensor[Float]].squeeze(1)
        }
      }
      else if (input.isTable) {
        val dataTable = input.toTable
        op match {
          case "add" => dataTable.keySet.foreach(key => {
            dataTable(key).asInstanceOf[Tensor[Float]].addSingletonDimension()
          })
          case _ => dataTable.keySet.foreach(key => {
            dataTable(key).asInstanceOf[Tensor[Float]].squeeze(1)
          })
        }
      }
    }
    input
  }

  /**
   * Use for single thread inference, to construct a batchSize = 1 input
   * Also return a Tensor if input Table has only one element
   * @param input Input table or tensor
   * @return input with single element batch constructed
   */
  def typeCheck(input: Activity): Activity = {
    if (input.isTable) {
      if (input.toTable.keySet.size == 1) {
        val key = input.toTable.keySet.head
        input.toTable(key).asInstanceOf[Tensor[Float]]
      }
      else {
        input.toTable
      }
    } else if (input.isTensor) {
      input.toTensor[Float]
    } else {
      logger.error("Your input of Inference is neither Table nor Tensor, please check.")
      throw new Error("Your input is invalid, skipped.")
    }
  }
}
