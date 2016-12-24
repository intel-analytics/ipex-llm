/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.lenet

import java.nio.file.Paths

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{GreyImgNormalizer, GreyImgToBatch, SampleToGreyImg}
import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{LocalValidator, Top1Accuracy, Validator}
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext

object Test {
  import Utils._

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).map(param => {
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark").map(conf => {
        conf.setAppName("Test Lenet on MNIST")
          .set("spark.akka.frameSize", 64.toString)
          .set("spark.task.maxFailures", "1")
        new SparkContext(conf)
      })

      val validationData = Paths.get(param.folder, "/t10k-images.idx3-ubyte")
      val validationLabel = Paths.get(param.folder, "/t10k-labels.idx1-ubyte")

      val validationSet = (if (sc.isDefined) {
        DataSet.array(load(validationData, validationLabel), sc.get, param.nodeNumber)
      } else {
        DataSet.array(load(validationData, validationLabel))
      }) -> SampleToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(
        param.batchSize)

      val model = Module.load[Float](param.model)
      Engine.setCoreNumber(param.coreNumber)
      val validator = Validator(model, validationSet)
      val result = validator.test(Array(new Top1Accuracy[Float]))
      result.foreach(r => {
        println(s"${r._2} is ${r._1}")
      })
    })
  }
}
