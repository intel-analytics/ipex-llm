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

import java.io.File
import java.security.{KeyStore, SecureRandom}
import java.util.concurrent.TimeUnit
import javax.net.ssl.{KeyManagerFactory, SSLContext, TrustManagerFactory}

import akka.actor.{ActorRef, ActorSystem, Props}
import akka.http.scaladsl.{ConnectionContext, Http}
import akka.http.scaladsl.server.Directives.{complete, path, _}
import akka.pattern.ask
import akka.stream.ActorMaterializer
import akka.util.Timeout
import com.codahale.metrics.{MetricRegistry, Timer}
import com.intel.analytics.bigdl.orca.inference.EncryptSupportive
import com.intel.analytics.bigdl.serving.utils.Conventions
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.concurrent.Await

object FrontEndApp extends Supportive with EncryptSupportive {
  override val logger = LoggerFactory.getLogger(getClass)

  val name = "BigDL web serving frontend"

  implicit val system = ActorSystem("bigdl-serving-frontend-system")
  implicit val materializer = ActorMaterializer()
  implicit val executionContext = system.dispatcher
  implicit val timeout: Timeout = Timeout(100, TimeUnit.SECONDS)

  def main(args: Array[String]): Unit = {
    timing(s"$name started successfully.")() {
      val arguments = timing("parse arguments")() {
        argumentsParser.parse(args, FrontEndAppArguments()) match {
          case Some(arguments) => logger.info(s"starting with $arguments"); arguments
          case None => argumentsParser.failure("miss args, please see the usage info"); null
        }
      }

      val servableManager = new ServableManager
      logger.info("Multi Serving Mode")
      timing("load servable manager")() {
        try servableManager.load(arguments.servableManagerPath, purePredictTimersMap,
          modelInferenceTimersMap)
        catch {
          case e: ServableLoadException =>
            throw e
          case e =>
            val exampleYaml =
              """
                ---
                 modelMetaDataList:
                 - !<ClusterServingMetaData>
                    modelName: "1"
                    modelVersion:"1.0"
                    redisHost: "localhost"
                    redisPort: "6381"
                    redisInputQueue: "serving_stream2"
                    redisOutputQueue: "cluster-serving_serving_stream2:"
                 - !<InflerenceModelMetaData>
                    modelName: "1"
                    modelVersion:"1.0"
                    modelPath:"/"
                    modelType:"OpenVINO"
                    features:
                      - "a"
                      - "b"
              """
            logger.info("Example Format of Input:" + exampleYaml)
            throw e
        }
      }
      logger.info("Servable Manager Load Success!")
      var redisPutter : ActorRef = null
      val route = timing("initialize http route")() {
        path("") {
          timing("welcome")(overallRequestTimer) {
            complete("welcome to " + name)
          }
        } ~ (get & path("metrics")) {
          timing("metrics")(overallRequestTimer, metricsRequestTimer) {
            val keys = metrics.getTimers().keySet()
            val servingMetrics = keys.toArray.map(key => {
              val timer = metrics.getTimers().get(key)
              ServingTimerMetrics(key.toString, timer)
            }).toList
            complete(jacksonJsonSerializer.serialize(servingMetrics))
          }
        } ~ (post & path("model-secure") &
          extract(_.request.entity.contentType) & entity(as[String])) {
          (contentType, content) => {
            try {
              if (redisPutter == null) {
                val redisPutterName = s"redis-putter"
                redisPutter = timing(s"$redisPutterName initialized.")() {
                  val redisPutterProps = Props(new RedisPutActor(
                    arguments.redisHost,
                    arguments.redisPort,
                    arguments.redisInputQueue,
                    arguments.redisOutputQueue,
                    arguments.timeWindow,
                    arguments.countWindow,
                    arguments.redisSecureEnabled,
                    arguments.redissTrustStorePath,
                    arguments.redissTrustStoreToken))
                  system.actorOf(redisPutterProps, name = redisPutterName)
                }
              }
              val secrets = content.split("&")
              val secret = secrets(0).split("=")(1)
              val salt = secrets(1).split("=")(1)
              val message = SecuredModelSecretSaltMessage(secret, salt)
              val result = Await.result(redisPutter ? message, timeout.duration)
                .asInstanceOf[Boolean]
              result match {
                case true => complete("model secured secrect and salt succeed to put in redis")
                case false => complete("model secured secrect and salt failed to put in redis")
              }
            } catch {
              case e: Exception =>
                e.printStackTrace()
                val error = ServingError(e.getMessage + "\n please post a content like " +
                  "secret=xxx&salt=xxxx")
                complete(500, error.toString)
            }

          }
        } ~ (get & path("models")) {
          timing("get all model infos")(overallRequestTimer, servablesRetriveTimer) {
            try {
              val servables = servableManager.retriveAllServables
              val metaData = servables.map(e => e.getMetaData)
              val json = JsonUtil.toJson(metaData)
              complete(200, json)
            }
            catch {
              case e: ModelNotFoundException =>
                complete(404, "Model Not Found")
              case e: ServingRuntimeException =>
                complete(405, "Serving Runtime Error Err: " + e)
              case e =>
                complete(500, "Internal Error: " + e)
            }
          }
        } ~ pathPrefix("models") {
          concat(
            (get & path(Segment)) {
              (modelName) => {
                timing("get model infos with model name")(overallRequestTimer,
                  servablesRetriveTimer) {
                  try {
                    val servables = servableManager.retriveServables(modelName)
                    val metaData = servables.map(e => e.getMetaData)
                    val json = JsonUtil.toJson(metaData)
                    complete(200, json)
                  }
                  catch {
                    case e: ModelNotFoundException =>
                      complete(404, "Model Not Found")
                    case e: ServingRuntimeException =>
                      complete(405, "Serving Runtime Error Err: " + e)
                    case e =>
                      complete(500, "Internal Error: " + e)
                  }
                }
              }
            } ~ (get & path(Segment / "versions" / Segment)) {
              (modelName, modelVersion) => {
                timing("get model info with model name and model version")(overallRequestTimer,
                  servableRetriveTimer) {
                  try {
                    val servables = servableManager.retriveServable(modelName, modelVersion)
                    val metaData = servables.getMetaData
                    val json = JsonUtil.toJson(metaData)
                    complete(200, json)
                  }
                  catch {
                    case e: ModelNotFoundException =>
                      complete(404, "Model Not Found")
                    case e: ServingRuntimeException =>
                      complete(405, "Serving Runtime Error Err: " + e)
                    case e =>
                      complete(500, "Internal Error: " + e)
                  }
                }
              }
            } ~ (post & path(Segment / "versions" / Segment / "predict")
              & extract(_.request.entity.contentType) & entity(as[String])) {
              (modelName, modelVersion, contentType, content) => {
                timing("backend inference")(overallRequestTimer, backendInferenceTimer) {
                  try {
                    logger.info("model name: " + modelName + ", model version: " + modelVersion)
                    val servable = timing("servable retrive")(servableRetriveTimer) {
                      servableManager.retriveServable(modelName, modelVersion)
                    }
                    val modelInferenceTimer = modelInferenceTimersMap(modelName)(modelVersion)
                    servable match {
                      case clusterServingServable: ClusterServingServable =>
                        val result = timing("cluster serving inference")(predictRequestTimer) {
                          val rejected = arguments.tokenBucketEnabled match {
                            case true =>
                              if (!clusterServingServable.rateLimiter.tryAcquire(
                                arguments.tokenAcquireTimeout, TimeUnit.MILLISECONDS)) {
                                true
                              } else {
                                false
                              }
                            case false => false
                          }
                          if (rejected) {
                            val error = ServingError("limited")
                            complete(500, error.toString)
                          }
                          val outputs = servable.getMetaData.
                            asInstanceOf[ClusterServingMetaData].inputCompileType match {
                            case "direct" =>
                              timing ("model inference direct") (modelInferenceTimer) {
                                servable.predict(content)
                              }
                            case "instance" =>
                              val instances = timing ("json deserialization") () {
                              JsonUtil.fromJson (classOf[Instances], content)
                              }
                              timing ("model inference") (modelInferenceTimer) {
                                servable.predict(instances)
                              }
                          }
                          Predictions(outputs)

                        }
                        timing("cluster serving response complete")() {
                          complete(200, result.toString)
                        }
                      case _: InferenceModelServable =>
                        val result = timing("inference model inference")(predictRequestTimer) {
                          val outputs = servable.getMetaData.
                            asInstanceOf[InferenceModelMetaData].inputCompileType match {
                            case "direct" => timing("model inference direct")(modelInferenceTimer) {
                              servable.predict(content)
                            }
                            case "instance" =>
                              val instances = timing("json deserialization")() {
                                JsonUtil.fromJson(classOf[Instances], content)
                              }
                              timing("model inference")(modelInferenceTimer) {
                                servable.predict(instances)
                              }
                          }
                          JsonUtil.toJson(outputs.map(_.result))
                        }
                        timing("inference model response complete")() {
                          complete(200, result)
                        }
                    }
                  }
                  catch {
                    case e: ModelNotFoundException =>
                      complete(404, "Model Not Found. Err: " + e.message)
                    case e: ServingRuntimeException =>
                      complete(405, "Serving Runtime Error Err: " + e.message)
                    case e =>
                      e.printStackTrace()
                      complete(500, "Internal Error: " + e)
                  }
                }
              }
            }
          )
        }
      }
      if (arguments.httpsEnabled) {
        val serverContext = defineServerContext(arguments.httpsKeyStoreToken,
          arguments.httpsKeyStorePath)
        Http().bindAndHandle(route, arguments.interface, port = arguments.securePort,
          connectionContext = serverContext)
        logger.info(s"https started at https://${arguments.interface}:${arguments.securePort}")
      }
      Http().bindAndHandle(route, arguments.interface, arguments.port)
      logger.info(s"http started at http://${arguments.interface}:${arguments.port}")
    }
  }


  val metrics = new MetricRegistry
  val overallRequestTimer = metrics.timer("bigdl.serving.request.overall")
  val predictRequestTimer = metrics.timer("bigdl.serving.request.predict")
  val servableRetriveTimer = metrics.timer("bigdl.serving.retrive.servable")
  val servablesRetriveTimer = metrics.timer("bigdl.serving.retrive.servables")
  val backendInferenceTimer = metrics.timer("bigdl.serving.backend.inference")
  val putRedisTimer = metrics.timer("bigdl.serving.redis.put")
  val getRedisTimer = metrics.timer("bigdl.serving.redis.get")
  val waitRedisTimer = metrics.timer("bigdl.serving.redis.wait")
  val metricsRequestTimer = metrics.timer("bigdl.serving.request.metrics")
  val modelInferenceTimersMap = new mutable.HashMap[String, mutable.HashMap[String, Timer]]
  val purePredictTimersMap = new mutable.HashMap[String, mutable.HashMap[String, Timer]]
  val makeActivityTimer = metrics.timer("bigdl.serving.activity.make")
  val handleResponseTimer = metrics.timer("bigdl.serving.response.handling")

  val jacksonJsonSerializer = new JacksonJsonSerializer()

  val argumentsParser = new scopt.OptionParser[FrontEndAppArguments]("AZ Serving") {
    head("BigDL Serving Frontend")
    opt[String]('i', "interface")
      .action((x, c) => c.copy(interface = x))
      .text("network interface of frontend")
    opt[Int]('p', "port")
      .action((x, c) => c.copy(port = x))
      .text("network port of frontend")
    opt[Int]('s', "securePort")
      .action((x, c) => c.copy(securePort = x))
      .text("https port of frontend")
    opt[String]('h', "redisHost")
      .action((x, c) => c.copy(redisHost = x))
      .text("host of redis")
    opt[Int]('r', "redisPort")
      .action((x, c) => c.copy(redisPort = x))
      .text("port of redis")
    opt[String]('i', "redisInputQueue")
      .action((x, c) => c.copy(redisInputQueue = x))
      .text("input queue of redis")
    opt[String]('o', "redisOutputQueue")
      .action((x, c) => c.copy(redisOutputQueue = x))
      .text("output queue  of redis")
    opt[Int]('l', "parallelism")
      .action((x, c) => c.copy(parallelism = x))
      .text("parallelism of frontend")
    opt[Int]('t', "timeWindow")
      .action((x, c) => c.copy(timeWindow = x))
      .text("timeWindow of frontend")
    opt[Int]('c', "countWindow")
      .action((x, c) => c.copy(countWindow = x))
      .text("countWindow of frontend")
    opt[Boolean]('e', "tokenBucketEnabled")
      .action((x, c) => c.copy(tokenBucketEnabled = x))
      .text("Token Bucket Enabled or not")
    opt[Int]('k', "tokensPerSecond")
      .action((x, c) => c.copy(tokensPerSecond = x))
      .text("tokens per second")
    opt[Int]('a', "tokenAcquireTimeout")
      .action((x, c) => c.copy(tokenAcquireTimeout = x))
      .text("token acquire timeout")
    opt[Boolean]('s', "httpsEnabled")
      .action((x, c) => c.copy(httpsEnabled = x))
      .text("https enabled or not")
    opt[String]('p', "httpsKeyStorePath")
      .action((x, c) => c.copy(httpsKeyStorePath = x))
      .text("https keyStore path")
    opt[String]('w', "httpsKeyStoreToken")
      .action((x, c) => c.copy(httpsKeyStoreToken = x))
      .text("https keyStore token")
    opt[Boolean]('s', "redisSecureEnabled")
      .action((x, c) => c.copy(redisSecureEnabled = x))
      .text("redis secure enabled or not")
    opt[Boolean]('s', "httpsEnabled")
      .action((x, c) => c.copy(httpsEnabled = x))
      .text("https enabled or not")
    opt[String]('p', "redissTrustStorePath")
      .action((x, c) => c.copy(redissTrustStorePath = x))
      .text("rediss trustStore path")
    opt[String]('w', "redissTrustStoreToken")
      .action((x, c) => c.copy(redissTrustStoreToken = x))
      .text("rediss trustStore password")
    opt[String]('z', "servableManagerConfPath")
      .action((x, c) => c.copy(servableManagerPath = x))
      .text("servableManagerConfPath")
  }

  def defineServerContext(httpsKeyStoreToken: String,
                          httpsKeyStorePath: String): ConnectionContext = {
    val token = httpsKeyStoreToken.toCharArray

    val keyStore = KeyStore.getInstance("PKCS12")
    val keystoreInputStream = new File(httpsKeyStorePath).toURI().toURL().openStream()
    require(keystoreInputStream != null, "Keystore required!")
    keyStore.load(keystoreInputStream, token)

    val keyManagerFactory = KeyManagerFactory.getInstance("SunX509")
    keyManagerFactory.init(keyStore, token)

    val trustManagerFactory = TrustManagerFactory.getInstance("SunX509")
    trustManagerFactory.init(keyStore)

    val sslContext = SSLContext.getInstance("TLS")
    sslContext.init(keyManagerFactory.getKeyManagers,
      trustManagerFactory.getTrustManagers, new SecureRandom)

    ConnectionContext.https(sslContext)
  }
}

case class FrontEndAppArguments(
                                 interface: String = "0.0.0.0",
                                 port: Int = 10020,
                                 securePort: Int = 10023,
                                 redisHost: String = "localhost",
                                 redisPort: Int = 6379,
                                 redisInputQueue: String = Conventions.SERVING_STREAM_DEFAULT_NAME,
                                 redisOutputQueue: String =
                                 Conventions.RESULT_PREFIX + Conventions.SERVING_STREAM_DEFAULT_NAME
                                   + ":",
                                 parallelism: Int = 1000,
                                 timeWindow: Int = 0,
                                 countWindow: Int = 0,
                                 tokenBucketEnabled: Boolean = false,
                                 tokensPerSecond: Int = 100,
                                 tokenAcquireTimeout: Int = 100,
                                 httpsEnabled: Boolean = false,
                                 httpsKeyStorePath: String = null,
                                 httpsKeyStoreToken: String = "1234qwer",
                                 redisSecureEnabled: Boolean = false,
                                 redissTrustStorePath: String = null,
                                 redissTrustStoreToken: String = "1234qwer",
                                 servableManagerPath: String = "./servables-conf.yaml"
                               )
