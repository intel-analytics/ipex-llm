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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.models.utils.{CachedModels, ModelBroadcast}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.intermediate.ConversionUtils
import com.intel.analytics.bigdl.utils.{Engine, MklDnn, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark
import org.apache.spark.SparkContext

object MemorySpec {
  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)

  private var sc: SparkContext = _


  def main(args: Array[String]): Unit = {
    System.setProperty("bigdl.check.singleton", false.toString)
    Engine.model.setPoolSize(1)
    Engine.setEngineType(MklDnn)

    sc = new spark.SparkContext(Engine.createSparkConf()
      .setMaster("local[1]")
      .setAppName("Memory leak checking")
        .set("spark.driver.memory", "4g")
    )
    Engine.init
    import com.intel.analytics.bigdl.models.resnet


    val module = resnet.ResNet(2, T("shortcutType" -> ShortcutType.B, "depth" -> 50,
      "optnet" -> false, "dataSet" -> DatasetType.ImageNet))
    val model = ConversionUtils.convert(module.evaluate()).evaluate()
    val bcast = ModelBroadcast[Float]().broadcast(sc, model)
    for(i <- 1 to 100) {
      val localModel = bcast.value(shareWeight = false)
      localModel.forward(Tensor[Float]( 3, 224, 224).rand())
      CachedModels.deleteKey[Float](bcast.uuid())
    }
    if (sc != null) {
      sc.stop()
    }
  }

}
