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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.nn.GradientChecker
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ModelGraientCheckSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "GoogleNet_v1 model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet_v1[Double](1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(model, input, 1e-2) should be(true)
  }

  "GoogleNet_v1 model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet_v1[Double](1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }

  "GoogleNet_v2 model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet_v2[Double](1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(model, input, 1e-2) should be(true)
  }

  "GoogleNet_v2 model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet_v2[Double](1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }

  "GoogleNet_v2_NoAuxClassifier model in batch mode" should "be good in " +
    "gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet_v2_NoAuxClassifier[Double](1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(model, input, 1e-2) should be(true)
  }

  "GoogleNet_v2_NoAuxClassifier model in batch mode" should "be good in " +
    "gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet_v2_NoAuxClassifier[Double](1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }

  "GoogleNet model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet.getModel[Double](1000, "googlenet")
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(model, input, 1e-2) should be(true)
  }

  "GoogleNet model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet.getModel[Double](1000, "googlenet")
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }

  "GoogleNet+bn model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet.getModel[Double](1000, "googlenet-bn")
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(model, input, 1e-2) should be(true)
  }

  "GoogleNet+bn model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = GoogleNet.getModel[Double](1000, "googlenet-bn")
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }

  "AlexNet-OWT model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](8, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = AlexNet_OWT[Double](1000, false, true)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-3)
    checker.checkLayer(model, input, 1e-2) should be(true)
  }

  "AlexNet-OWT model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](8, 3, 224, 224).apply1(e => RNG.uniform(0, 1))
    val model = AlexNet_OWT[Double](1000, false, true)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }

  "AlexNet model in batch mode" should "be good in graident check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](8, 3, 227, 227).apply1(e => RNG.uniform(0, 1))
    val model = AlexNet[Double](1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-3)
    checker.checkLayer(model, input, 1e-2) should be(true)
  }

  "AlexNet model in batch mode" should "be good in graident check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](8, 3, 227, 227).apply1(e => RNG.uniform(0, 1))
    val model = AlexNet[Double](1000)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-3)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }

  "VggLike model in batch mode" should "be good in gradient check for input" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](8, 3, 32, 32).apply1(e => RNG.uniform(0, 1))
    val model = VggLike[Double](10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkLayer(model, input, 1e-2) should be(true)
  }

  "VggLike model in batch mode" should "be good in gradient check for weight" in {
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](8, 3, 32, 32).apply1(e => RNG.uniform(0, 1))
    val model = VggLike[Double](10)
    model.zeroGradParameters()

    val checker = new GradientChecker(1e-4)
    checker.checkWeight(model, input, 1e-2) should be(true)
  }
}
