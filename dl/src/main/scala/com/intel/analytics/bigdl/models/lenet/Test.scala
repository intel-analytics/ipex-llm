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

import com.intel.analytics.bigdl.nn.Module
import com.intel.analytics.bigdl.optim.{Top1Accuracy, LocalValidator}

object Test {
  import Options._

  def main(args: Array[String]): Unit = {
    val batchSize = 32
    testParser.parse(args, new TestParams()).map(param => {
      val validationData = Paths.get(param.folder, "/t10k-images.idx3-ubyte")
      val validationLabel = Paths.get(param.folder, "/t10k-labels.idx1-ubyte")
      val validateDataSet = DataSet.localDataSet(validationData, validationLabel, false, batchSize)
      val model = Module.load[Float](param.model)
      val validator = new LocalValidator[Float](model, param.coreNumber)
      val result = validator.test(validateDataSet, Array(new Top1Accuracy[Float]))
      result.foreach(r => {
        println(s"${r._2} is ${r._1}")
      })

    })
  }
}
