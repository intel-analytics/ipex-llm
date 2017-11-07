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

package com.intel.analytics.bigdl.models.inception

import com.intel.analytics.bigdl.dataset.{ByteRecord, DataSet}
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Top1Accuracy, Top5Accuracy, Validator}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.hadoop.io.Text
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)


  import Options._

  val imageSize = 224

  def main(args: Array[String]) {
    testParser.parse(args, new TestParams()).foreach { param =>
      val batchSize = param.batchSize.getOrElse(128)
      val conf = Engine.createSparkConf().setAppName("Test Inception on ImageNet")
      val sc = new SparkContext(conf)
      Engine.init

      // We set partition number to be node*core, actually you can also assign other partitionNum
      val partitionNum = Engine.nodeNumber() * Engine.coreNumber()
      val rawData = sc.sequenceFile(param.folder, classOf[Text], classOf[Text], partitionNum)
        .map(image => {
          ByteRecord(image._2.copyBytes(), DataSet.SeqFileFolder.readLabel(image._1).toFloat)
        }).coalesce(partitionNum, true)

      val rddData = DataSet.SeqFileFolder.filesToRdd(param.folder, sc, 1000)
      val transformer = BytesToBGRImg() -> BGRImgCropper(imageSize, imageSize, CropCenter) ->
        HFlip(0.5) -> BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) -> BGRImgToSample()
      val evaluationSet = transformer(rddData)

      val model = Module.load[Float](param.model)
      val result = model.evaluate(evaluationSet,
        Array(new Top1Accuracy[Float], new Top5Accuracy[Float]), param.batchSize)

      result.foreach(r => println(s"${r._2} is ${r._1}"))
      sc.stop()
    }
  }
}
