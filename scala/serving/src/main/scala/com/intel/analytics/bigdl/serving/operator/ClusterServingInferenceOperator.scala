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

package com.intel.analytics.bigdl.serving.operator

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.serving.{ClusterServing, ClusterServingHelper, ClusterServingInference}
import com.intel.analytics.bigdl.serving.utils.Conventions
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.logging.log4j.{LogManager, Logger}


class ClusterServingInferenceOperator(var params: ClusterServingParams = new ClusterServingParams())
  extends RichMapFunction[List[(String, Activity)], List[(String, String)]] {
  var logger: Logger = null
  var inference: ClusterServingInference = null
  ClusterServing.helper = new ClusterServingHelper()

  override def open(parameters: Configuration): Unit = {
    logger = LogManager.getLogger(getClass)
    if (params == null) {
      params = new ClusterServingParams()
    }
    if (ClusterServing.model == null) {
      ClusterServing.synchronized {
        if (ClusterServing.model == null) {
          logger.info("Loading Cluster Serving model...")
          val localModelDir = getRuntimeContext.getDistributedCache
            .getFile(Conventions.SERVING_MODEL_TMP_DIR).getPath
          val info = ClusterServingHelper
            .loadModelfromDir(localModelDir, params._modelConcurrent)
          ClusterServing.model = info._1
          params._modelType = info._2
          ClusterServing.helper.modelType = info._2
        }
      }
    }
    inference = new ClusterServingInference()
  }

  /**
   * To inference input, user has to construct Activity first
   * To construct Activity, use ClusterServingInput, e.g.
   * A single dimension input [1,2,3]
   * ClusterServingInput("my-input", Array(1,2,3))
   * A 2 dimension input [[1,2], [3,4], [5,6]]
   * ClusterServingInput("my-input", Array(1,2,3,4), Array(3,2))
   * A multiple input [1], [[1,2],[3,4]]
   * Format is ClusterServingInput(name, Array(valueArray: Array[Float], ShapeArray: Array[Int]))
   * ClusterServingInput("my-input", Array((Array(1), Array(1)),
   *                                       (Array(1,2,3,4), Array(2,2))
   *                                       ))
   * A String type input ["my", "input"]
   * ClusterServingInput("my-string-input", Array("my", "input"))
   * @param in The List of input where each element is a tuple (name: String, input: Activity)
   * @return
   */
  override def map(in: List[(String, Activity)]): List[(String, String)] = {
    val t1 = System.nanoTime()
    val postProcessed =
      inference.singleThreadInference(in).toList
    val t2 = System.nanoTime()
    logger.info(s"${postProcessed.size} records backend time ${(t2 - t1) / 1e9} s. " +
      s"Throughput ${postProcessed.size / ((t2 - t1) / 1e9)}")
    postProcessed
  }
}


