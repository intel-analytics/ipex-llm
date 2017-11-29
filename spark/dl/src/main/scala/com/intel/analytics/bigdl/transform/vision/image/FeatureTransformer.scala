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

import com.intel.analytics.bigdl.dataset.{ChainedTransformer, Transformer}
import com.intel.analytics.bigdl.opencv.OpenCV
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.log4j.Logger

abstract class FeatureTransformer() extends Transformer[ImageFeature, ImageFeature] {

  import FeatureTransformer.logger

  private var outKey: Option[String] = None

  def setOutKey(key: String): this.type = {
    outKey = Some(key)
    this
  }

  protected def transformMat(feature: ImageFeature): Unit = {}

  def transform(feature: ImageFeature): ImageFeature = {
    require(OpenCV.isOpenCVLoaded, "opencv isn't loaded")
    if (!feature.isValid) return feature
    try {
      transformMat(feature)
      if (outKey.isDefined) {
        require(outKey.get != ImageFeature.mat, s"the output key should not equal to" +
          s" ${ImageFeature.mat}, please give another name")
        if (feature.contains(outKey.get)) {
          val mat = feature[OpenCVMat](outKey.get)
          feature.opencvMat().copyTo(mat)
        } else {
          feature(outKey.get) = feature.opencvMat().clone()
        }
      }
    } catch {
      case e: Exception =>
        val path = if (feature.contains(ImageFeature.uri)) feature(ImageFeature.uri) else ""
        logger.warn(s"failed ${path} in transformer ${getClass}")
        e.printStackTrace()
        feature.isValid = false
    }
    feature
  }

  override def apply(prev: Iterator[ImageFeature]): Iterator[ImageFeature] = {
    prev.map(transform)
  }

  def apply(imageFrame: ImageFrame): ImageFrame = {
    imageFrame.transform(this)
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  def -> (other: FeatureTransformer): FeatureTransformer = {
    new ChainedFeatureTransformer(this, other)
  }

  // scalastyle:off methodName
  // scalastyle:off noSpaceBeforeLeftBracket
  override def -> [C](other: Transformer[ImageFeature, C]): Transformer[ImageFeature, C] = {
    new ChainedTransformer(this, other)
  }
}

object FeatureTransformer {
  val logger = Logger.getLogger(getClass)
}


class ChainedFeatureTransformer(first: FeatureTransformer, last: FeatureTransformer) extends
  FeatureTransformer {

  override def transform(prev: ImageFeature): ImageFeature = {
    last.transform(first.transform(prev))
  }
}
