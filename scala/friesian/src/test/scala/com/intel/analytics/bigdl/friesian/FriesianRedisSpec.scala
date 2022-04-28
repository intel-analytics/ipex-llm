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

package com.intel.analytics.bigdl.friesian

import com.intel.analytics.bigdl.friesian.serving.feature.utils.{LettuceUtils, RedisType}
import org.scalatest.Ignore

import java.util

@Ignore
class FriesianRedisSpec extends ZooSpecHelper {
  "Lettuce utils standalone 0 replica" should "work properly" in {
    val hostPort = new util.ArrayList[(String, Integer)]()
    hostPort.add(("localhost", 6379))
    val utils = LettuceUtils.getInstance(RedisType.STANDALONE, hostPort, "",
      null, null, 0)
    val data = Array(Array("standalone0a", "a"), Array("standalone0b", "b"))
    utils.MSet("", data)
    val keys = Array("a", "2tower_user", "standalone0a", "standalone0b")
    val result = utils.MGet("", keys)
    TestUtils.conditionFailTest(result.size() == 4)
    TestUtils.conditionFailTest(result.get(0) == "" && result.get(2) == "a"
      && result.get(3) == "b")
  }

  "Lettuce utils standalone 1 replica" should "work properly" in {
    val hostPort = new util.ArrayList[(String, Integer)]()
    hostPort.add(("localhost", 6379))
    hostPort.add(("localhost", 6380))
    val utils = LettuceUtils.getInstance(RedisType.STANDALONE, hostPort, "1re",
      null, null, 0)
    val data = Array(Array("standalone1a", "1rea"), Array("standalone1b", "1reb"))
    utils.MSet("", data)
    val keys = Array("a", "2tower_user", "standalone1a", "standalone1b")
    val result = utils.MGet("", keys)
    TestUtils.conditionFailTest(result.size() == 4)
    TestUtils.conditionFailTest(
      result.get(0) == "" && result.get(2) == "1rea" && result.get(3) == "1reb")
  }

  "Lettuce utils sentinel 1 replica" should "work properly" in {
    val utils = LettuceUtils.getInstance(RedisType.SENTINEL, null, "1sre",
      "localhost:26379", "mymaster", 0)
    val data = Array(Array("sentinel1a", "1srea"), Array("sentinel1b", "1sreb"))
    utils.MSet("", data)
    val keys = Array("a", "2tower_user", "sentinel1a", "sentinel1b")
    val result = utils.MGet("", keys)
    TestUtils.conditionFailTest(result.size() == 4)
    TestUtils.conditionFailTest(
      result.get(0) == "" && result.get(2) == "1srea" && result.get(3) == "1sreb")
  }

  "Lettuce utils cluster 0" should "work properly" in {
    val hostPort = new util.ArrayList[(String, Integer)]()
    hostPort.add(("localhost", 12000))
    hostPort.add(("localhost", 12001))
    hostPort.add(("localhost", 12002))
    val utils = LettuceUtils.getInstance(RedisType.CLUSTER, hostPort, "cluster",
      null, null, 0)
    val data = Array(Array("clustera", "clua"), Array("clusterb", "club"))
    utils.MSet("", data)
    val keys = Array("a", "2tower_user", "clustera", "clusterb")
    val result = utils.MGet("", keys)
    TestUtils.conditionFailTest(result.size() == 4)
    TestUtils.conditionFailTest(
      result.get(0) == "" && result.get(2) == "clua" && result.get(3) == "club")
  }

  "Lettuce utils cluster 1" should "work properly" in {
    val hostPort = new util.ArrayList[(String, Integer)]()
    hostPort.add(("localhost", 12000))
    hostPort.add(("localhost", 12001))
    hostPort.add(("localhost", 12002))
    val utils = LettuceUtils.getInstance(RedisType.CLUSTER, hostPort, "cluster",
      null, null, 1)
    val data = Array(Array("1", "11"), Array("2", "12"), Array("3", "13"), Array("4", "14"),
      Array("5", "15"), Array("6", "16"), Array("7", "17"), Array("8", "18"), Array("9", "19"),
      Array("10", "110"))
    utils.MSet("item", data)
    val keys = Array("a", "2tower_user", "1", "2")
    val result = utils.MGet("item", keys)
    TestUtils.conditionFailTest(result.size() == 4)
    TestUtils.conditionFailTest(
      result.get(0) == "" && result.get(2) == "11" && result.get(3) == "12")
  }

  "Lettuce utils cluster 2" should "work properly" in {
    val hostPort = new util.ArrayList[(String, Integer)]()
    hostPort.add(("localhost", 12000))
    hostPort.add(("localhost", 12001))
    hostPort.add(("localhost", 12002))
    val utils = LettuceUtils.getInstance(RedisType.CLUSTER, hostPort, "cluster",
      null, null, 2)
    val data = Array(Array("1", "21"), Array("2", "22"), Array("3", "23"), Array("4", "24"),
      Array("5", "25"), Array("6", "26"), Array("7", "27"), Array("8", "28"), Array("9", "29"),
      Array("10", "210"), Array("64", "24"))
    utils.MSet("item", data)
    utils.MSet("user2", data)
    val keys = Array("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
    val result = utils.MGet("item", keys)
    TestUtils.conditionFailTest(result.size() == 10)
    TestUtils.conditionFailTest(
      result.get(0) == "21" && result.get(2) == "23" && result.get(3) == "24")
  }
}
