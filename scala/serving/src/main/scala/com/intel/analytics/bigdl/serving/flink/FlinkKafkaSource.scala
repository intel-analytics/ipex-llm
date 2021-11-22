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

package com.intel.analytics.bigdl.serving.flink

import java.time.Duration
import java.util.{Collections, Properties}

import com.intel.analytics.bigdl.serving.{ClusterServing, ClusterServingHelper}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.source.{RichParallelSourceFunction, SourceFunction}
import org.apache.kafka.clients.consumer.{ConsumerConfig, ConsumerRecords, KafkaConsumer}
import org.apache.kafka.common.TopicPartition
import org.apache.log4j.Logger
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.collection.JavaConverters._


class FlinkKafkaSource()
  extends RichParallelSourceFunction[List[(String, String, String)]] {
  @volatile var isRunning = true
  var logger: Logger = null
  var consumer: KafkaConsumer[String, String] = null
  var helper: ClusterServingHelper = null
  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)
    helper = ClusterServing.helper
    val props = new Properties()
    props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, helper.kafkaUrl)
    props.put(ConsumerConfig.GROUP_ID_CONFIG, "serving")
    props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG,
      "org.apache.kafka.common.serialization.StringDeserializer")
    props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG,
      "org.apache.kafka.common.serialization.StringDeserializer")

    consumer = new KafkaConsumer[String, String](props)
    consumer.subscribe(Collections.singletonList(helper.jobName))
  }

  override def run(sourceContext: SourceFunction
  .SourceContext[List[(String, String, String)]]): Unit = while (isRunning) {
    implicit val formats = DefaultFormats
    val records: ConsumerRecords[String, String] = consumer.poll(Duration.ofMillis(1))
    if (records != null) {
      val messages = records.records(new TopicPartition(helper.jobName, 0))
      if (messages != null) {
        messages.asScala.foreach(message => {
          val parsedValue = parse(message.value()).extract[Map[String, String]]
          sourceContext.collect(
            List(
              (parsedValue.getOrElse("uri", null),
                parsedValue.getOrElse("data", null),
                parsedValue.getOrElse("serde", null))
            )
          )
        })
      }
    }
  }

  override def cancel(): Unit = {
    consumer.close()
    logger.info("Flink source cancelled")
  }
}
