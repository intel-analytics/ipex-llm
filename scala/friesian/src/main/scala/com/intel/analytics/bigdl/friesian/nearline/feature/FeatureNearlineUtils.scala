package com.intel.analytics.bigdl.friesian.nearline.feature

import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils
import com.intel.analytics.bigdl.friesian.serving.feature.utils.RedisUtils
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils.objToBytes
import org.apache.logging.log4j.{LogManager, Logger}

import java.util.{Base64, List => JList}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.col

import scala.collection.JavaConverters._

object FeatureNearlineUtils {
  val logger: Logger = LogManager.getLogger(getClass)

  def loadUserItemFeaturesRDD(spark: SparkSession, redis: RedisUtils): Unit = {
    assert(NearlineUtils.helper.initialUserDataPath != null ||
      NearlineUtils.helper.initialItemDataPath != null, "initialUserDataPath or " +
      "initialItemDataPath should be provided if loadInitialData is true")
    if (NearlineUtils.helper.initialUserDataPath != null) {
      assert(NearlineUtils.helper.userIDColumn != null)
      assert(NearlineUtils.helper.userFeatureColArr != null)
      logger.info("Start inserting user features...")
      val colNames = NearlineUtils.helper.userFeatureColArr.mkString(",")
      redis.setSchema("user", colNames)
      val userFeatureColumns = NearlineUtils.helper.userIDColumn +: NearlineUtils.helper.userFeatureColArr
      divideFileAndLoad(spark, NearlineUtils.helper.initialUserDataPath, userFeatureColumns,
        "user")
    }

    if (NearlineUtils.helper.initialItemDataPath != null) {
      assert(NearlineUtils.helper.itemIDColumn != null)
      assert(NearlineUtils.helper.itemFeatureColArr != null)
      logger.info("Start inserting item features...")
      val colNames = NearlineUtils.helper.itemFeatureColArr.mkString(",")
      redis.setSchema("item", colNames)
      val itemFeatureColumns = NearlineUtils.helper.itemIDColumn +: NearlineUtils.helper.itemFeatureColArr
      divideFileAndLoad(spark, NearlineUtils.helper.initialItemDataPath, itemFeatureColumns,
        "item")
    }
    logger.info(s"Insert finished")
  }

  def divideFileAndLoad(spark: SparkSession, dataDir: String, featureCols: Array[String],
                        keyPrefix: String): Unit = {
    var totalCnt: Long = 0
    val readList = NearlineUtils.getListOfFiles(dataDir)
    val start = System.currentTimeMillis()
    for (parquetFiles <- readList) {
      var df = spark.read.parquet(parquetFiles: _*)
      df = df.select(featureCols.map(col): _*).distinct()
      val cnt = df.count()
      totalCnt = totalCnt + cnt
      logger.info(s"Load ${cnt} features into redis.")
      val featureRDD = df.rdd.map(row => {
        encodeRow(row)
      })
      featureRDD.foreachPartition { partition =>
        if (partition.nonEmpty) {
          val redis = RedisUtils.getInstance(256,
            NearlineUtils.helper.redisHostPort,
            NearlineUtils.helper.redisKeyPrefix, NearlineUtils.helper.itemSlotType)
          redis.Mset(keyPrefix, partition.toArray)
        }
      }
    }
    val end = System.currentTimeMillis()
    logger.info(s"Insert ${totalCnt} features into redis, takes: ${(end - start) / 1000}s")
  }

  def encodeRow(row: Row): JList[String] = {
    val id = row.get(0).toString
    val rowArr = row.toSeq.drop(1).toArray
    val encodedValue = java.util.Base64.getEncoder.encodeToString(objToBytes(rowArr))
    List(id, encodedValue).asJava
  }

  @deprecated
  def encodeRowWithCols(row: Row, cols: Array[String]): JList[String] = {
    val rowSeq = row.toSeq
    val id = rowSeq.head.toString
    val colValueMap = (cols zip rowSeq).toMap
    val encodedValue = java.util.Base64.getEncoder.encodeToString(
      objToBytes(colValueMap))
    List(id, encodedValue).asJava
  }
}
