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

package com.intel.analytics.sparkdl.utils

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}


class FileSpec extends FlatSpec with Matchers {
  "save/load Java object file" should "work properly" in {

    val tmpFile = java.io.File.createTempFile("module", "obj")
    val absolutePath = tmpFile.getAbsolutePath


    val module = new Sequential[Double]

    module.add(new SpatialConvolution(1, 6, 5, 5))
    module.add(new Tanh())
    module.add(new SpatialMaxPooling(2, 2, 2, 2))
    // stage 2 : filter bank -> squashing -> max pooling
    module.add(new SpatialConvolution(6, 12, 5, 5))
    module.add(new Tanh())
    module.add(new SpatialMaxPooling(2, 2, 2, 2))
    //      -- stage 3 : standard 2-layer neural network
    module.add(new Reshape(Array(12 * 5 * 5)))
    module.add(new Linear(12 * 5 * 5, 100))
    module.add(new Tanh())
    module.add(new Linear(100, 6))
    module.add(new LogSoftMax[Double]())

    Tensor.saveObj(module, absolutePath, true)
    val testModule: Module[Double] = Tensor.loadObj(absolutePath)

    testModule should be(module)
  }

}
