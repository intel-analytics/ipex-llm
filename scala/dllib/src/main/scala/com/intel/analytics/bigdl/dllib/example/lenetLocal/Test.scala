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

package com.intel.analytics.bigdl.dllib.example.lenetLocal

import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, SampleToBatch}
import com.intel.analytics.bigdl.dllib.feature.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToSample}
import com.intel.analytics.bigdl.dllib.nn.Module
import com.intel.analytics.bigdl.dllib.optim.{Top1Accuracy, ValidationMethod}
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator

object Test {
  Configurator.setLevel("org", Level.ERROR)
  Configurator.setLevel("akka", Level.ERROR)
  Configurator.setLevel("breeze", Level.ERROR)


  import Utils._

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).foreach { param =>
      System.setProperty("bigdl.localMode", "true")
      System.setProperty("bigdl.coreNumber", param.coreNumber.toString)
      Engine.init

      val validationData = param.folder + "/t10k-images-idx3-ubyte"
      val validationLabel = param.folder + "/t10k-labels-idx1-ubyte"

      val evaluationSet = DataSet.array(load(validationData, validationLabel)) ->
        BytesToGreyImg(28, 28) ->
        GreyImgNormalizer(trainMean, trainStd) ->
        GreyImgToSample() -> SampleToBatch(
        batchSize = param.batchSize, None, None, None,
        partitionNum = Some(1))

      val model = Module.loadModule[Float](param.model)
      val result = model.evaluate(evaluationSet.toLocal(),
        Array(new Top1Accuracy[Float].asInstanceOf[ValidationMethod[Float]]))
      result.foreach(r => println(s"${r._2} is ${r._1}"))
    }
  }
}
