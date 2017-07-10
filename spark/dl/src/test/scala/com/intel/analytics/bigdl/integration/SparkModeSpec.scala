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

package com.intel.analytics.bigdl.integration

import com.intel.analytics.bigdl.models.lenet
import com.intel.analytics.bigdl.models.vgg
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Integration
class SparkModeSpec extends FlatSpec with Matchers with BeforeAndAfter{

  val mnistFolder = System.getProperty("mnist")
  val cifarFolder = System.getProperty("cifar")

  "Lenet model train and validate" should "be correct" in {
    val batchSize = 8
    val args = Array("--folder", mnistFolder, "-b", batchSize.toString, "-e", "1")
    lenet.Train.main(args)
  }

  "Vgg model train and validate" should "be correct" in {
    val batchSize = 8
    val args = Array("--folder", cifarFolder, "-b", batchSize.toString, "-e", "1")
    vgg.Train.main(args)
  }
}
