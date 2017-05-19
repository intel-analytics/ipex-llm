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
package com.intel.analytics.bigdl.example.finetune_flickr_style

import java.nio.file.{Path, Paths}

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.dataset.image._
import com.intel.analytics.bigdl.DataSet
import org.apache.spark.SparkContext

object FlickrImage {
  def load(
             path: Path,
             sc: Option[SparkContext],
             imageSize: Int,
             batchSize: Int
           )
  : DataSet[MiniBatch[Float]] = {
    (if (sc.isDefined) {
      DataSet.SeqFileFolder.files(path.toString(),
        sc.get, classNum = 20) ->
        BytesToBGRImg(normalize = 1f) // do not normalize the pixel values to [0, 1]
    } else {
      DataSet.ImageFolder.paths(path) -> LocalImgReader(256, 256, normalize = 1f)
    }) -> BGRImgCropper(imageSize, imageSize, CropCenter) ->
      BGRImgNormalizer(123, 117, 104, 1, 1, 1) ->
      BGRImgToBatch(batchSize, toRGB = false)
  }

}
