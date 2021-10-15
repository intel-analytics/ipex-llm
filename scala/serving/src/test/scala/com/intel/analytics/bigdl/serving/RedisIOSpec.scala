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

package com.intel.analytics.bigdl.serving

import java.util
import java.util.AbstractMap.SimpleEntry

import com.intel.analytics.bigdl.serving.pipeline.{RedisEmbeddedReImpl, RedisUtils}
import com.intel.analytics.bigdl.serving.http.Supportive
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}
import redis.clients.jedis.{Jedis, ScanParams, StreamEntryID}
import redis.embedded.RedisExecProvider
import redis.embedded.util.OS
import scopt.OptionParser

import scala.collection.JavaConverters._

class RedisIOSpec(path : String) extends FlatSpec
  with Matchers with BeforeAndAfter with Supportive {
  val redisHost = "localhost"
  val redisPort = 6379
  val pathToRedisExecutable = path
  var redisServer: RedisEmbeddedReImpl = _
  var jedis: Jedis = _

  val inputHash = List("index1" -> "data1", "index2" -> "data2")
  val inputXStream = Map("name1" -> "A", "name2" -> "B").asJava

  before {
    val customProvider = RedisExecProvider.defaultProvider.`override`(OS.UNIX,
      pathToRedisExecutable)
    redisServer = new RedisEmbeddedReImpl(customProvider, redisPort)
    redisServer.start()

    jedis = new Jedis(redisHost, redisPort)
  }

  after {
    redisServer.stop()
  }

  "redisServer" should "works well" in {
    redisServer shouldNot be (null)
    jedis shouldNot be (null)
  }

  "redisServer" should "have correct output" in {
    xgroupCreate()
    xstreamWrite(inputXStream, "test")
    readRedis()
    hashmapWrite(inputHash)
    readRedis()
  }

  def xgroupCreate() : Unit = {
    println("Create Group <xstream>\n")
    try {
      jedis.xgroupCreate("test",
        "xstream", new StreamEntryID(0, 0), true)
    } catch {
      case e: Exception =>
        logger.info(s"xgroupCreate raise [$e], " +
          s"will not create new group.")
    }
  }

  def readRedis() : Unit = {
    println("Read group <xstream>")
    val response = jedis.xreadGroup(
      "xstream",
      "fake consumer",
      10,
      10,
      false,
      new SimpleEntry("test", StreamEntryID.UNRECEIVED_ENTRY))
    println(response)

    println("Read all keys")
    val params = new ScanParams
    params.`match`("*")

    // Use "0" to do a full iteration of the collection.
    val scanResult = jedis.scan("0", params)
    val keys = scanResult.getResult
    println(keys)
  }

  def hashmapWrite(value: List[(String, String)]) : Unit = {
    println("Write Hash Map to Redis")
    val ppl = jedis.pipelined()
    var cnt = 0
    value.foreach(v => {
      RedisUtils.writeHashMap(ppl, v._1, v._2, "test")
      if (v._2 != "NaN") {
        cnt += 1
      }
    })
    ppl.sync()
    logger.info(s"${cnt} valid records written to redis")
  }

  def xstreamWrite(hash: util.Map[String, String], streamID: String) : Unit = {
    println(s"Write to Redis stream ${streamID}")
    val ppl = jedis.pipelined()
    ppl.xadd(streamID, StreamEntryID.NEW_ENTRY, hash)
    ppl.sync()
    logger.info(s"${hash.size()} valid records written to redis")
  }

}

// If scalatest could not run successfully,
// uncomment this part to run the test

object RedisIOTest {
  // initialize the parser
  case class Config(path: String = null)
  val parser = new OptionParser[Config]("RedisIO test Usage") {
    opt[String]('p', "path")
      .text("Path to Redis Server Executable")
      .action((x, params) => params.copy(path = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    val arg = parser.parse(args, Config()).head
    val path = arg.path

    val redisIOMockInstance = new RedisIOSpec(path)
    redisIOMockInstance.execute()
  }

}


