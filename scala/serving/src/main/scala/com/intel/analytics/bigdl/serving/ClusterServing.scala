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

import com.intel.analytics.bigdl.orca.inference.InferenceModel
import com.intel.analytics.bigdl.serving.flink.{FlinkInference, FlinkKafkaSink, FlinkKafkaSource, FlinkRedisSink, FlinkRedisSource}
import com.intel.analytics.bigdl.serving.utils.{ConfigParser, Conventions, RedisUtils}
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.log4j.Logger
import redis.clients.jedis.{JedisPool, JedisPoolConfig}
import scopt.OptionParser


object ClusterServing {
  case class ServingParams(configPath: String = "config.yaml",
                           timerMode: Boolean = false)
  val logger = Logger.getLogger(getClass)
  var argv: ServingParams = _
  var helper: ClusterServingHelper = _
  var streamingEnv: StreamExecutionEnvironment = _
  var model: InferenceModel = _
  var jedisPool: JedisPool = _
  val jedisPoolConfig = new JedisPoolConfig()
  jedisPoolConfig.setMaxTotal(256)
  val parser = new OptionParser[ServingParams]("Text Classification Example") {
    opt[String]('c', "configPath")
      .text("Config Path of Cluster Serving")
      .action((x, params) => params.copy(configPath = x))
    opt[Boolean]("timerMode")
      .text("Whether to open timer mode")
      .action((x, params) => params.copy(timerMode = x))
  }
  def uploadModel(): Unit = {
    streamingEnv = StreamExecutionEnvironment.getExecutionEnvironment
    streamingEnv.registerCachedFile(helper.modelPath, Conventions.SERVING_MODEL_TMP_DIR)
    if (helper.redisSecureEnabled) {
      streamingEnv.registerCachedFile(helper.redisSecureTrustStorePath, Conventions.SECURE_TMP_DIR)
    }
  }
  def executeJob(): Unit = {
    /**
     * Flink environment parallelism depends on model parallelism
     */
    // Uncomment this line if you need to check predict time in debug
    // Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.DEBUG)
    streamingEnv.setParallelism(helper.modelParallelism)
    if (helper.queueUsed == "kafka") {
      streamingEnv.addSource(new FlinkKafkaSource())
        .map(new FlinkInference())
        .addSink(new FlinkKafkaSink(helper))
    } else {
      streamingEnv.addSource(new FlinkRedisSource())
        .map(new FlinkInference())
        .addSink(new FlinkRedisSink(helper))
    }


    logger.info(s"Cluster Serving Flink job graph details \n${streamingEnv.getExecutionPlan}")
    streamingEnv.executeAsync()
  }
  def initializeRedis(): Unit = {
    val params = ClusterServing.helper
    if (params.redisSecureEnabled) {
      System.setProperty("javax.net.ssl.trustStore", params.redisSecureTrustStorePath)
      System.setProperty("javax.net.ssl.trustStorePassword", params.redisSecureTrustStorePassword)
      System.setProperty("javax.net.ssl.keyStoreType", "JKS")
      System.setProperty("javax.net.ssl.keyStore", params.redisSecureTrustStorePath)
      System.setProperty("javax.net.ssl.keyStorePassword", params.redisSecureTrustStorePassword)
    }
    if (jedisPool == null) {
      this.synchronized {
        if (jedisPool == null) {
          logger.info(
            s"Creating JedisPool at ${params.redisHost}:${params.redisPort}")
          val jedisPoolConfig = new JedisPoolConfig()
          jedisPoolConfig.setMaxTotal(256)
          jedisPool = new JedisPool(jedisPoolConfig,
            params.redisHost, params.redisPort, params.redisTimeout, params.redisSecureEnabled)
        }
      }
    }

    logger.info(
      s"FlinkRedisSource connect to Redis: redis://${params.redisHost}:${params.redisPort} " +
        s"with timeout: ${params.redisTimeout} and redisSecureEnabled: " +
        s"${params.redisSecureEnabled}")
    params.redisSecureEnabled match {
      case true => logger.info(
        s"FlinkRedisSource connect to secured Redis successfully.")
      case false => logger.info(
        s"FlinkRedisSource connect to plain Redis successfully.")
    }

    //    // add Redis configuration here if necessary
    val jedis = RedisUtils.getRedisClient(jedisPool)
    jedis.close()
  }
  def main(args: Array[String]): Unit = {
    argv = parser.parse(args, ServingParams()).head
    val configParser = new ConfigParser(argv.configPath)
    helper = configParser.loadConfig()
    helper.configPath = argv.configPath
    uploadModel()
    executeJob()
  }
}
