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

package com.intel.analytics.bigdl.transform.vision.image

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class ConvertorSpec extends FlatSpec with Matchers {
  val resource = getClass.getClassLoader.getResource("pascal/")

  "MatToTensor" should "work properly" in {
    val data = ImageFrame.read(resource.getFile)
    val imF = data.asInstanceOf[LocalImageFrame].array.head
    val tensor2 = imF.toTensor(ImageFeature.floats)
    val transformer = MatToTensor[Float]()
    transformer(data)
    val tensor = imF[Tensor[Float]](ImageFeature.imageTensor)
    tensor should be (tensor2)
  }

}
