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

package com.intel.analytics.bigdl.serving.engine

import java.util.AbstractMap.SimpleEntry
import java.util.UUID

import com.intel.analytics.bigdl.serving.ClusterServing
import com.intel.analytics.bigdl.serving.pipeline.RedisUtils
import com.intel.analytics.bigdl.serving.utils.{ClusterServingHelper, Conventions}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.source.{RichParallelSourceFunction, RichSourceFunction, SourceFunction}
import org.apache.log4j.Logger
import redis.clients.jedis.{Jedis, JedisPool, JedisPoolConfig, StreamEntryID}

import scala.collection.JavaConverters._

class FlinkRedisSource()
  extends RichParallelSourceFunction[List[(String, String, String)]] {
  @volatile var isRunning = true
  var jedis: Jedis = null
  var logger: Logger = null
  var helper: ClusterServingHelper = null
  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)
    helper = ClusterServing.helper
    ClusterServing.initializeRedis()
    jedis = RedisUtils.getRedisClient(ClusterServing.jedisPool)
    RedisUtils.createRedisGroupIfNotExist(jedis, helper.jobName)
  }

  override def run(sourceContext: SourceFunction
    .SourceContext[List[(String, String, String)]]): Unit = while (isRunning) {
    val groupName = "serving"
    val consumerName = "consumer-" + UUID.randomUUID().toString
    val readNumPerTime = if (helper.modelType == "openvino") helper.threadPerModel else 1

    val response = jedis.xreadGroup(
      groupName,
      consumerName,
      readNumPerTime,
      1,
      false,
      new SimpleEntry(helper.jobName, StreamEntryID.UNRECEIVED_ENTRY))
    if (response != null) {
      for (streamMessages <- response.asScala) {
        val key = streamMessages.getKey
        val entries = streamMessages.getValue.asScala
        val it = entries.map(e => {
          (e.getFields.get("uri"), e.getFields.get("data"), e.getFields.get("serde"))
        }).toList
        sourceContext.collect(it)
      }
      RedisUtils.checkMemory(jedis, 0.6, 0.5)
    }
  }

  override def cancel(): Unit = {
    jedis.close()
    logger.info("Flink source cancelled")
  }

}
