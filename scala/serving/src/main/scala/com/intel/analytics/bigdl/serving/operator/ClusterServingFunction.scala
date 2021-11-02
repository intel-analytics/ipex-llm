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

import java.nio.file.Files

import com.intel.analytics.bigdl.serving.{ClusterServing, ClusterServingHelper, ClusterServingInference}
import com.intel.analytics.bigdl.serving.serialization.ArrowDeserializer
import org.apache.flink.core.fs.Path
import org.apache.flink.table.functions.{FunctionContext, ScalarFunction}
import org.apache.flink.util.FileUtils
import org.slf4j.LoggerFactory

@SerialVersionUID(1L)
class ClusterServingFunction()
  extends ScalarFunction {
  val clusterServingParams = new ClusterServingParams()
  ClusterServing.helper = new ClusterServingHelper()
  var inference: ClusterServingInference = null
  val logger = LoggerFactory.getLogger(getClass)

  def copyFileToLocal(modelPath: String): String = {
    val localModelPath =
      Files.createTempDirectory("cluster-serving").toFile.toString + "/model"
    logger.info(s"Copying model from $modelPath to local $localModelPath")
    FileUtils.copy(new Path(modelPath), new Path(localModelPath), false)
    logger.info("model copied")
    localModelPath
  }

  override def open(context: FunctionContext): Unit = {
    val modelPath = context.getJobParameter("modelPath", "")
    require(modelPath != "", "You have not provide modelPath in job parameter.")
    val modelLocalPath = copyFileToLocal(modelPath)
    if (ClusterServing.model == null) {
      ClusterServing.synchronized {
        if (ClusterServing.model == null) {
          logger.info("Loading Cluster Serving model...")
          val info = ClusterServingHelper
            .loadModelfromDir(modelLocalPath, clusterServingParams._modelConcurrent)
          ClusterServing.jobModelMap += (modelPath -> info._1)
          ClusterServing.model = info._1
          clusterServingParams._modelType = info._2
          ClusterServing.helper.modelType = clusterServingParams._modelType
        }
      }
    }
    inference = new ClusterServingInference(modelPath)

  }

  def eval(uri: String, data: String): String = {
    val array = data.split(" +").map(_.toFloat)
    val input = ClusterServingInput(uri, array)
    val result = inference.singleThreadInference(List((uri, input)))
    ArrowDeserializer(result.head._2)
  }
}
