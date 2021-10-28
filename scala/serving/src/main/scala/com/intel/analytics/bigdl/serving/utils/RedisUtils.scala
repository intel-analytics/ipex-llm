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

package com.intel.analytics.bigdl.serving.utils

import org.apache.log4j.Logger
import redis.clients.jedis.exceptions.JedisConnectionException
import redis.clients.jedis.{Jedis, JedisPool, Pipeline, StreamEntryID}

import scala.collection.JavaConverters._

object RedisUtils {
  val logger = Logger.getLogger(getClass)
  def createRedisGroupIfNotExist(jedis: Jedis, streamName: String): Unit = {
    try {
      jedis.xgroupCreate(streamName,
        "serving", new StreamEntryID(0, 0), true)
    } catch {
      case e: Exception =>
        logger.info(s"xgroupCreate raise [$e], " +
          s"will not create new group.")
    }
  }
  def checkMemory(db: Jedis, inputThreshold: Double, cutRatio: Double): Unit = {
    var redisInfo = RedisUtils.getMapFromInfo(db.info())
    if (redisInfo("used_memory").toLong >=
      redisInfo("maxmemory").toLong * inputThreshold) {
      this.synchronized {
        redisInfo = RedisUtils.getMapFromInfo(db.info())
        if (redisInfo("maxmemory").toLong > 0 && redisInfo("used_memory").toLong >=
          redisInfo("maxmemory").toLong * inputThreshold) {
          logger.warn(s"Used memory ${redisInfo("used_memory")}, " +
            s"Max memory ${redisInfo("maxmemory")}. Your input data length is " +
            s"${db.xlen(Conventions.SERVING_STREAM_DEFAULT_NAME)}. Removing old data...")
          db.xtrim(Conventions.SERVING_STREAM_DEFAULT_NAME,
            (db.xlen(Conventions.SERVING_STREAM_DEFAULT_NAME) * cutRatio).toLong, true)
          logger.warn(s"Trimmed stream, now your serving stream length is " +
            s"${db.xlen(Conventions.SERVING_STREAM_DEFAULT_NAME)}")
          var cuttedRedisInfo = RedisUtils.getMapFromInfo(db.info())
          while (cuttedRedisInfo("used_memory").toLong >=
            cuttedRedisInfo("maxmemory").toLong * inputThreshold) {
            logger.error(s"Used memory ${redisInfo("used_memory")}, " +
              s"Max memory ${redisInfo("maxmemory")}. " +
              s"Your result field has exceeded the limit, please dequeue. Will retry in 10 sec..")
            cuttedRedisInfo = RedisUtils.getMapFromInfo(db.info())
            Thread.sleep(10000)
          }
        }
      }
    }
  }


  def getMapFromInfo(info: String): Map[String, String] = {
    var infoMap = Map[String, String]()
    val tabs = info.split("#")

    for (tab <- tabs) {
      if (tab.length > 0) {
        val keys = tab.split("\r\n")

        for (key <- keys) {
          if (key.split(":").size == 2) {
            infoMap += (key.split(":").head ->
              key.split(":").last)
          }
        }
      }
    }

    return infoMap
  }
  def getRedisClient(redisPool: JedisPool): Jedis = {
    var jedis: Jedis = null
    var cnt: Int = 0
    while (jedis == null) {
      try {
        jedis = redisPool.getResource
      }
      catch {
        case e: JedisConnectionException =>
          logger.info(
            s"Redis client can not connect, maybe max number of clients is reached." +
            "Waiting, if you always receive this, please stop your service and report bug.")
          e.printStackTrace()
          cnt += 1
          if (cnt >= 10) {
            throw new Error("can not get redis from the pool")
          }
          Thread.sleep(500)
      }
      Thread.sleep(10)
    }
    jedis
  }
  def writeHashMap(ppl: Pipeline, key: String, value: String, name: String): Unit = {
    val hKey = Conventions.RESULT_PREFIX + name + ":" + key
    val hValue = Map[String, String]("value" -> value).asJava
    ppl.hmset(hKey, hValue)
  }
  def writeXstream(ppl: Pipeline, key: String, value: String, name: String): Unit = {
    val streamKey = Conventions.RESULT_PREFIX + name + ":" + key
    val streamValue = Map[String, String]("value" -> value).asJava
    ppl.xadd(streamKey, StreamEntryID.NEW_ENTRY, streamValue)
  }

}
