/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.example.ImageClassification

import java.nio.file.Path

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, _}
import com.intel.analytics.bigdl.dataset.{Transformer, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File

object Preprocessor {
  def apply(path: Path, imageSize: Int, batchSize: Int,
    transformers: Transformer[LocalLabeledImagePath, LabeledBGRImage], hasLabel: Boolean = true)
  : DataSet[MiniBatch[Float]] = {
    val paths = if (hasLabel) {
      DataSet.ImageFolder.paths(path)
    } else {
      DataSet.ImageFolder.pathsNoLabel(path)
    }
    paths.transform(
      MTLabeledBGRImgToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = transformers, toRGB = false))
  }
}

object AlexNetPreprocessor {
  val imageSize = 227

  def transformers(meanFile: String)
  : Transformer[LocalLabeledImagePath, LabeledBGRImage] = {
    val means = File.load[Tensor[Float]](meanFile)
    (LocalImgReader(256, 256, normalize = 1f) -> BGRImgPixelNormalizer(means)
      -> BGRImgCropper(imageSize, imageSize, CropCenter))
  }

  def apply(path: Path, batchSize: Int, meanFile: String, hasLabel: Boolean = true)
  : DataSet[MiniBatch[Float]] = {
    Preprocessor(path, imageSize, batchSize, transformers(meanFile), hasLabel)
  }
}

object InceptionPreprocessor {
  val imageSize = 224

  def transformers: Transformer[LocalLabeledImagePath, LabeledBGRImage]
  = (LocalImgReader(256, 256, normalize = 1f)
    -> BGRImgCropper(imageSize, imageSize, CropCenter)
    -> BGRImgNormalizer(123, 117, 104, 1, 1, 1))

  def apply(path: Path, batchSize: Int, hasLabel: Boolean = true)
  : DataSet[MiniBatch[Float]] = {
    Preprocessor(path, imageSize, batchSize, transformers, hasLabel)
  }
}

object ResNetPreprocessor {
  val imageSize = 224

  def apply(path: Path, batchSize: Int, hasLabel: Boolean = true)
  : DataSet[MiniBatch[Float]] = {
    val paths = if (hasLabel) {
      DataSet.ImageFolder.paths(path)
    } else {
      DataSet.ImageFolder.pathsNoLabel(path)
    }
    paths.transform(
      MTLabeledBGRImgToBatch(
        width = imageSize,
        height = imageSize,
        batchSize = batchSize,
        transformer = LocalImgReader(256) ->
          BGRImgCropper(cropWidth = imageSize, cropHeight = imageSize, CropCenter) ->
          BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225)

      )
    )
  }
}

