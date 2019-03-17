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

package com.intel.analytics.bigdl.example.int8

import scopt.OptionParser

object Utils {
  case class TestParams(
    folder: String = "./",
    model: String = "",
    batchSize: Int = 128,
    quantize: Boolean = false
  )

  val testParser = new OptionParser[TestParams]("BigDL ResNet on Cifar10 Test Example") {
    opt[String]('f', "folder")
      .text("the location of Cifar10 dataset")
      .action((x, c) => c.copy(folder = x))

    opt[String]('m', "model")
      .text("the location of model snapshot")
      .action((x, c) => c.copy(model = x))
      .required()
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Boolean]('q', "quantize")
      .text("quantize the model")
      .action((x, c) => c.copy(quantize = x))
  }
}
