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

package com.intel.analytics.bigdl.friesian.nearline.utils

import scala.beans.BeanProperty

class NearlineHelper extends Serializable {
  // feature service attributes
  @BeanProperty var initialUserDataPath: String = _
  @BeanProperty var initialItemDataPath: String = _
  @BeanProperty var userFeatureColumns: String = _
  @BeanProperty var itemFeatureColumns: String = _
  @BeanProperty var userIDColumn: String = _
  @BeanProperty var redisKeyPrefix: String = _
  @BeanProperty var redisClusterItemSlotType = 0
  @BeanProperty var redisUrl = "localhost:6379"

  // recall service attributes
  @BeanProperty var indexPath: String = _
  @BeanProperty var initialDataPath: String = _
  @BeanProperty var indexDim: Int = 128
  @BeanProperty var part: Int = 20
  @BeanProperty var itemEmbeddingColumn: String = _

  // feature & recall service attributes
  @BeanProperty var userModelPath: String = _
  @BeanProperty var itemModelPath: String = _
  @BeanProperty var itemIDColumn: String = _

  var redisHostPort: java.util.ArrayList[(String, Integer)] =
    new java.util.ArrayList[(String, Integer)]()
  var itemSlotType: Int = 0
  var userFeatureColArr: Array[String] = _
  var itemFeatureColArr: Array[String] = _

  def parseConfigStrings(): Unit = {
    redisUrl.split("\\s*,\\s*").foreach(url => {
      val redisHost = url.split(":").head.trim
      val redisPort = url.split(":").last.trim.toInt
      redisHostPort.add(Tuple2(redisHost, redisPort))
    })
    if (redisKeyPrefix == null) {
      redisKeyPrefix = ""
    }

    if (userFeatureColumns != null) {
      userFeatureColArr = userFeatureColumns.split("\\s*,\\s*")
    }
    if (itemFeatureColumns != null) {
      itemFeatureColArr = itemFeatureColumns.split("\\s*,\\s*")
    }

    itemSlotType = if (redisClusterItemSlotType != 0 && redisClusterItemSlotType != 1 &&
      redisClusterItemSlotType != 2) {
      0
    } else {
      redisClusterItemSlotType
    }
  }
}
