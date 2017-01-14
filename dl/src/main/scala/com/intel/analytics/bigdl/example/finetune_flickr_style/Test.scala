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
package com.intel.analytics.bigdl.example.finetune_flickr_style

import java.nio.file.Paths

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.spark.SparkContext

object Test {

  import Options._

  def main(args: Array[String]): Unit = {
    testParser.parse(args, new TestParams()).map(param => {
      val sc = Engine.init(param.nodeNumber, param.coreNumber, param.env == "spark")
        .map(conf => {
          conf.setAppName("BigDL Flickr Style fine tune Example")
            .set("spark.task.maxFailures", "1").setMaster("local")
          new SparkContext(conf)
        })
      val imageSize = 227
      // create dataset
      val validationData = Paths.get(param.folder, "test")
      val validateDataSet = FlickrImage.load(validationData, sc, imageSize, param.batchSize)
      // Create model
      val model = {
        Module.load[Float](param.model)
      }

      val validator = Validator(model, validateDataSet)
      val result = validator.test(Array(new Top1Accuracy[Float], new Top5Accuracy[Float]))
      result.foreach(r => {
        println(s"${r._2} is ${r._1}")
      })
    })
  }
}
