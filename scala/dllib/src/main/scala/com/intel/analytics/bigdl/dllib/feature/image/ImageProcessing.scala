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

package com.intel.analytics.bigdl.dllib.feature.image

import com.intel.analytics.bigdl.dllib.feature.transform.vision.image.{FeatureTransformer, ImageFeature}
import com.intel.analytics.bigdl.dllib.feature.common.{ChainedPreprocessing, Preprocessing}


abstract class ImageProcessing extends FeatureTransformer with
   Preprocessing[ImageFeature, ImageFeature] {
  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (other: ImageProcessing): ImageProcessing = {
    new ChainedImageProcessing(this, other)
  }
  // scalastyle:on noSpaceBeforeLeftBracket
  // scalastyle:on methodName
}

/**
 * A transformer chain two ImageProcessing together.
 */
class ChainedImageProcessing(first: ImageProcessing, last: ImageProcessing) extends
  ImageProcessing {

  override def transform(prev: ImageFeature): ImageFeature = {
    last.transform(first.transform(prev))
  }
}
