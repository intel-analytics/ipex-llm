/*
 * Copyright 2021 The BigDL Authors.
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

import java.time.Duration
import java.util.{Collections, Properties}

import com.intel.analytics.bigdl.serving.utils.ClusterServingHelper
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.source.{RichParallelSourceFunction, SourceFunction}
import org.apache.kafka.clients.consumer.{ConsumerConfig, ConsumerRecords, KafkaConsumer}
import org.apache.kafka.common.TopicPartition
import org.apache.log4j.Logger

import scala.collection.JavaConverters._


class FlinkKafkaSource(params: ClusterServingHelper)
  extends RichParallelSourceFunction[List[(String, String, String)]] {
  @volatile var isRunning = true
  var logger: Logger = null
  var consumer: KafkaConsumer[String, String] = null

  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)

    val props = new Properties()
    props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092")
    props.put(ConsumerConfig.GROUP_ID_CONFIG, "serving")
    props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG,
      "org.apache.kafka.common.serialization.StringDeserializer")
    props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG,
      "org.apache.kafka.common.serialization.StringDeserializer")

    consumer = new KafkaConsumer[String, String](props)
    consumer.subscribe(Collections.singletonList(params.jobName))
  }

  override def run(sourceContext: SourceFunction
  .SourceContext[List[(String, String, String)]]): Unit = while (isRunning) {
    val records: ConsumerRecords[String, String] = consumer.poll(Duration.ofMillis(1))
    if (records != null) {
      val messages = records.records(new TopicPartition(params.jobName, 0))
      if (messages != null) {
        messages.asScala.foreach(message => {
          sourceContext.collect(
            List((message.key(), message.value(), "serde"))
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
