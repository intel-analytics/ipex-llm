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

package com.intel.analytics.bigdl.serving.http

import java.util
import java.util.{HashMap, UUID}

import akka.actor.{Actor, ActorRef}
import com.intel.analytics.bigdl.serving.utils.{Conventions, RedisUtils}
import org.slf4j.LoggerFactory
import redis.clients.jedis.JedisPool

import scala.collection.JavaConverters._
import scala.collection.mutable

class RedisIOActor(redisOutputQueue: String = Conventions.RESULT_PREFIX +
  Conventions.SERVING_STREAM_DEFAULT_NAME + ":",
                   redisInputQueue: String = "serving_stream",
                   jedisPool: JedisPool = null) extends Actor with Supportive {
  override val logger = LoggerFactory.getLogger(getClass)
  val jedis = if (jedisPool == null) {
    RedisUtils.getRedisClient(new JedisPool())
  } else {
    RedisUtils.getRedisClient(jedisPool)
  }
  var requestMap = mutable.Map[String, ActorRef]()

  override def receive: Receive = {
    case message: DataInputMessage =>
      silent(s"${self.path.name} input message process")() {
        logger.info(s"${System.currentTimeMillis()} Input enqueue ${message.id} at time ")
        enqueue(redisInputQueue, message)

        requestMap += (redisOutputQueue + message.id -> sender())
      }
    case message: DequeueMessage =>
      if (!requestMap.isEmpty) {
        dequeue(redisOutputQueue).foreach(result => {
          logger.info(s"${System.currentTimeMillis()} Get redis result at time ")
          val queryOption = requestMap.get(result._1)
          if (queryOption != None) {
            val queryResult = result._2.asScala
            queryOption.get ! ModelOutputMessage(queryResult)
            requestMap -= result._1
            logger.info(s"${System.currentTimeMillis()} Send ${result._1} back at time ")
          }
        })
      }

  }
  def enqueue(queue: String, input: DataInputMessage): Unit = {
    timing(s"${self.path.name} put request to redis")(FrontEndApp.putRedisTimer) {
      val hash = new HashMap[String, String]()
//      val bytes = StreamSerializer.objToBytes(input.inputs)
//      val b64 = java.util.Base64.getEncoder.encodeToString(bytes)
      hash.put("uri", input.id)
      hash.put("data", input.inputs)
      hash.put("serde", "stream")
      jedis.xadd(queue, null, hash)
    }
  }
  def dequeue(queue: String): mutable.Set[(String, util.Map[String, String])] = {
    val resultSet = jedis.keys(s"${queue}*")
    val res = resultSet.asScala.map(key => {
      (key, jedis.hgetAll(key))

    })
    resultSet.asScala.foreach(key => jedis.del(key))
    res
  }

}
