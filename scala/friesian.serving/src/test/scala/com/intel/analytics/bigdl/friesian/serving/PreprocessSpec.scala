package com.intel.analytics.bigdl.friesian.serving

import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features
import com.intel.analytics.bigdl.friesian.serving.utils.Utils
import com.intel.analytics.bigdl.grpc.ConfigParser
import com.intel.analytics.bigdl.friesian.serving.utils.gRPCHelper

import java.net.URL
import utils.recommender.RecommenderUtils

import scala.collection.JavaConverters._

class PreprocessSpec extends ZooSpecHelper {
  val resource: URL = getClass.getClassLoader.getResource("")

  "Preprocess" should "work properly" in {
    val userCols = List("enaging_user_follower_count", "enaging_user_following_count",
      "enaging_user_is_verified")
    val itemCols = List("present_media", "language", "tweet_type",
      "engaged_with_user_follower_count", "engaged_with_user_following_count", "len_hashtags",
      "len_domains", "len_links", "present_media_language", "engaged_with_user_is_verified")
    val userFeature = Features.newBuilder().addID(14).addB64Feature("rO0ABXVyABNbT" +
      "GphdmEubGFuZy5PYmplY3Q7kM5YnxBzKWwCAAB4cAAAAANzcgARamF2YS5sYW5nLkludGVnZXIS4qCk94GHOAIAAUk" +
      "ABXZhbHVleHIAEGphdmEubGFuZy5OdW1iZXKGrJUdC5TgiwIAAHhwAAAABHNxAH4AAgAAAANzcQB+AAIAAAAA")
      .addAllColNames(userCols.asJava).build()
    val itemIds: List[Integer] = List.fill(1000)(232)
    val itemValues = List.fill(1000)("rO0ABXVyABNbTGphdmEubGFuZy5PYmplY3Q7kM5YnxBzKWwCAAB4cAAAAAp" +
      "zcgARamF2YS5sYW5nLkludGVnZXIS4qCk94GHOAIAAUkABXZhbHVleHIAEGphdmEubGFuZy5OdW1iZXKGrJUdC5Tgi" +
      "wIAAHhwAAAABXNxAH4AAgAAACtzcQB+AAIAAAACc3EAfgACAAAABnNxAH4AAgAAAAFzcgAPamF2YS5sYW5nLkZsb2F" +
      "02u3Jots88OwCAAFGAAV2YWx1ZXhxAH4AAzyj1wpzcQB+AAkAAAAAc3EAfgAJAAAAAHNxAH4AAgAAARlxAH4ACA==")
    val itemFeature = Features.newBuilder().addAllID(itemIds.asJava).addAllColNames(itemCols.asJava)
      .addAllB64Feature(itemValues.asJava).build()
    Utils.helper = new gRPCHelper
    Utils.helper.inferenceColArr = Array("engaged_with_user_is_verified",
      "enaging_user_is_verified", "present_media_language", "present_media", "tweet_type",
      "language", "engaged_with_user_follower_count", "engaged_with_user_following_count",
      "enaging_user_follower_count", "enaging_user_following_count", "len_hashtags", "len_domains",
      "len_links")
    for (i <- 0.until(1000)) {
      RecommenderUtils.featuresToRankingInputSet(userFeature, itemFeature, 0)
    }
  }
}
