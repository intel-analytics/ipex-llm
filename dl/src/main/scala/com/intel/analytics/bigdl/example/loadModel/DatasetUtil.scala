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

package com.intel.analytics.bigdl.example.loadModel

import java.nio.file.Path

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dataset.image.{BGRImgCropper, BGRImgNormalizer, _}
import com.intel.analytics.bigdl.dataset.{Transformer, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.File
import org.apache.spark.SparkContext

object Preprocessor {
  def apply(path: Path, batchSize: Int,
    transformers: Transformer[LabeledBGRImage, LabeledBGRImage],
    sc: Option[SparkContext] = None, nodeNumber: Int = -1)
  : DataSet[MiniBatch[Float]] = {
    (if (sc.isDefined) {
      DataSet.ImageFolder.images(path, sc.get, nodeNumber, 256, 256, 1f)
    } else {
      DataSet.ImageFolder.images(path, 256, 256, 1f)
    }) -> transformers -> BGRImgToBatch(batchSize, toRGB = false)
  }

  object AlexNetPreprocessor {
    val imageSize = 227

    def apply(path: Path, batchSize: Int, meanFile: String, sc: Option[SparkContext] = None,
      nodeNumber: Int = -1, hasLabel: Boolean = true)
    : DataSet[MiniBatch[Float]] = {
      val means = File.load[Tensor[Float]](meanFile)
      val transformers = BGRImgPixelNormalizer(means) ->
        BGRImgCropper(imageSize, imageSize, CropCenter)
      Preprocessor(path, batchSize, transformers, sc, nodeNumber)
    }
  }

  object InceptionPreprocessor {
    val imageSize = 224

    def apply(path: Path, batchSize: Int, sc: Option[SparkContext] = None, nodeNumber: Int = -1)
    : DataSet[MiniBatch[Float]] = {
      val transformers = (BGRImgCropper(imageSize, imageSize, CropCenter)
        -> BGRImgNormalizer(123, 117, 104, 1, 1, 1))
      Preprocessor(path, batchSize, transformers, sc, nodeNumber)
    }
  }

  object ResNetPreprocessor {
    val imageSize = 224

    def apply(path: Path, batchSize: Int, sc: Option[SparkContext] = None, nodeNumber: Int = -1)
    : DataSet[MiniBatch[Float]] = {
      (if (sc.isDefined) {
        DataSet.ImageFolder.images(path, sc.get, nodeNumber, 256)
      } else {
        DataSet.ImageFolder.images(path, 256)
      }) -> BGRImgCropper(cropWidth = imageSize, cropHeight = imageSize, CropCenter) ->
        BGRImgNormalizer(0.485, 0.456, 0.406, 0.229, 0.224, 0.225) ->
        BGRImgToBatch(batchSize)
    }
  }

}

