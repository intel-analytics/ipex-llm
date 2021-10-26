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

import java.util.Properties

import com.intel.analytics.bigdl.serving.ClusterServing
import com.intel.analytics.bigdl.serving.utils.{ClusterServingHelper, Conventions}
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.sink.{RichSinkFunction, SinkFunction}
import org.apache.kafka.clients.producer.{KafkaProducer, ProducerConfig, ProducerRecord}
import org.apache.log4j.Logger

class FlinkKafkaSink(params: ClusterServingHelper)
  extends RichSinkFunction[List[(String, String)]] {
  var producer: KafkaProducer[String, String] = null
  var logger: Logger = null
  var topic: String = null


  override def open(parameters: Configuration): Unit = {
    logger = Logger.getLogger(getClass)
    ClusterServing.helper = params
    topic = Conventions.RESULT_PREFIX + params.jobName

    val props = new Properties()
    props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, params.kafkaUrl)

    props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG,
      "org.apache.kafka.common.serialization.StringSerializer")
    props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG,
      "org.apache.kafka.common.serialization.StringSerializer")

    producer = new KafkaProducer[String, String](props)
  }

  override def close(): Unit = {
    if (null != producer) {
      producer.close()
    }
  }

  override def invoke(value: List[(String, String)], context: SinkFunction.Context[_]): Unit = {
    var cnt = 0
    value.foreach(v => {
      val record = new ProducerRecord(topic, v._1, v._2)
      producer.send(record)
      if (v._2 != "NaN") {
        cnt += 1
      }
    })
    logger.info(s"${cnt} valid records written to Kafka")
  }

}
