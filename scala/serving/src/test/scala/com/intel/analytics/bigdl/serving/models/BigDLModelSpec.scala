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

package com.intel.analytics.bigdl.serving.models

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.serving.{ClusterServing, ClusterServingHelper}
import org.scalatest.{FlatSpec, Matchers}
import scala.sys.process._

class BigDLModelSpec extends FlatSpec with Matchers {
  "BigDL NNFrame from keras model" should "work" in {
    ("wget --no-check-certificate -O /tmp/linear.model https://sourceforge.net/" +
      "projects/analytics-zoo/files/analytics-zoo-data/linear.model").!
    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "bigdl"
    helper.weightPath = "/tmp/linear.model"
    ClusterServing.model = helper.loadInferenceModel()
    "rm /tmp/linear.model".!
    val tensor = Tensor[Float](1, 2).rand()
    val result = ClusterServing.model.doPredict(tensor)
    require(result.toTensor[Float].size().sameElements(Array(1, 1)), "shape error")
  }

  "BigDL NNframe from nn model" should "work" in {
    ("wget --no-check-certificate -O /tmp/bigdl-nnframe-resnet-50.model https://sourceforge.net/" +
      "projects/analytics-zoo/files/analytics-zoo_resnet-50_imagenet_0.1.0.model").!
    ClusterServing.helper = new ClusterServingHelper()
    val helper = ClusterServing.helper
    helper.modelType = "bigdl"
    helper.weightPath = "/tmp/bigdl-nnframe-resnet-50.model"
    ClusterServing.model = helper.loadInferenceModel()
    val tensor = Tensor[Float](1, 3, 224, 224).rand()
    val result = ClusterServing.model.doPredict(tensor)
    require(result.toTensor[Float].size().sameElements(Array(1000)), "shape error")
  }
}
