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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.dmg.pmml.False
import org.scalatest.{FlatSpec, Matchers}

class BoxHeadSpec extends FlatSpec with Matchers {
  "BoxHead" should "be ok" in {
    val inChannels: Int = 6
    val resolution: Int = 7
    val scales: Array[Float] = Array[Float](0.25f, 0.125f)
    val samplingRratio: Float = 2.0f
    val scoreThresh: Float = 0.05f
    val nmsThresh: Float = 0.5f
    val detections_per_img: Int = 100
    val representation_size: Int = 1024
    val numClasses: Int = 81 // coco dataset class number

    val layer = new BoxHead(inChannels, resolution, scales, samplingRratio, scoreThresh,
      nmsThresh, detections_per_img, representation_size, numClasses)

//    val params = layer.getParameters()
//    params._1.fill(0.001f)

    val features1 = Tensor[Float](T(T(T(T(0.5381, 0.0856, 0.1124, 0.7493),
      T(0.4624, 0.2182, 0.7364, 0.3522),
      T(0.7552, 0.7117, 0.2715, 0.9082)),
      T(T(0.0928, 0.2735, 0.7539, 0.7539),
        T(0.4777, 0.1525, 0.8279, 0.6481),
        T(0.6019, 0.4803, 0.5869, 0.7459)),
      T(T(0.1924, 0.2795, 0.4463, 0.3887),
        T(0.5791, 0.9832, 0.8752, 0.4598),
        T(0.2278, 0.0758, 0.4988, 0.3742)),
      T(T(0.1762, 0.6499, 0.2534, 0.9842),
        T(0.0908, 0.8676, 0.1700, 0.1887),
        T(0.7138, 0.9559, 0.0119, 0.7799)),
      T(T(0.8200, 0.6767, 0.3637, 0.9771),
        T(0.1217, 0.5645, 0.2574, 0.6729),
        T(0.6140, 0.5333, 0.4425, 0.1740)),
      T(T(0.3994, 0.9148, 0.0123, 0.0125),
        T(0.5663, 0.9951, 0.8143, 0.9906),
        T(0.0923, 0.8285, 0.2992, 0.2221)))))

    val features2 = Tensor[Float](T(T(T(T(0.0492, 0.1234),
      T(0.3291, 0.0613),
      T(0.4260, 0.1422),
      T(0.2282, 0.4258),
      T(0.7426, 0.9476)),
      T(T(0.6662, 0.7015),
        T(0.4598, 0.6378),
        T(0.9571, 0.4947),
        T(0.1659, 0.3034),
        T(0.8583, 0.1369)),
      T(T(0.1711, 0.6440),
        T(0.2099, 0.4468),
        T(0.9518, 0.3877),
        T(0.4058, 0.6630),
        T(0.9056, 0.4054)),
      T(T(0.4562, 0.0277),
        T(0.2358, 0.3938),
        T(0.9187, 0.4067),
        T(0.0445, 0.4171),
        T(0.3434, 0.1964)),
      T(T(0.9473, 0.7239),
        T(0.1732, 0.5352),
        T(0.8276, 0.6435),
        T(0.3516, 0.3760),
        T(0.3437, 0.0198)),
      T(T(0.7811, 0.5682),
        T(0.5121, 0.9655),
        T(0.3496, 0.7632),
        T(0.4267, 0.4533),
        T(0.8624, 0.3172)))))

    val bbox = Tensor[Float](T(T(1.0f, 3.0f, 2.0f, 6.0f),
      T(3.0f, 5.0f, 6.0f, 10.0f)))
    val labels = Tensor[Float](T(1, 3))

    val output = layer.forward(T(T(features1, features2), bbox, labels)).toTable

    val expectOutput = Tensor[Float]()


  }
}
