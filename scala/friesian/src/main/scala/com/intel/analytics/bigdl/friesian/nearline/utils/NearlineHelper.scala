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
