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

package com.intel.analytics.bigdl.serving.example

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.serving.operator.{ClusterServingInferenceOperator, ClusterServingInput}
import com.intel.analytics.bigdl.serving.serialization.ArrowDeserializer
import com.intel.analytics.bigdl.serving.utils.Conventions
import org.apache.flink.streaming.api.functions.sink.{RichSinkFunction, SinkFunction}
import org.apache.flink.streaming.api.functions.source.{RichParallelSourceFunction, SourceFunction}
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator
import scopt.OptionParser

object ClusterServingOperatorUsageExample {
  case class ExampleParams(modelPath: String = null)
  val parser = new OptionParser[ExampleParams]("Cluster Serving Operator Usage Example") {
    opt[String]('m', "modelPath")
      .text("Model Path of Cluster Serving")
      .action((x, params) => params.copy(modelPath = x))
      .required()
  }
  Configurator.setLevel("org", Level.ERROR)
  def main(args: Array[String]): Unit = {
    val arg = parser.parse(args, ExampleParams()).head

    val serving = StreamExecutionEnvironment.getExecutionEnvironment
    // Use Flink distributed cache to copy model to every executors
    serving.registerCachedFile(arg.modelPath, Conventions.SERVING_MODEL_TMP_DIR)
    serving.addSource(new MySource()).setParallelism(1)
        .map(new ClusterServingInferenceOperator()).setParallelism(1)
        .addSink(new MySink()).setParallelism(1)
    serving.execute()
  }

}
class MySource extends RichParallelSourceFunction[List[(String, Activity)]] {
  var cnt = 0
  @volatile var isRunning = true
  override def run(ctx: SourceFunction.SourceContext[List[(String, Activity)]]): Unit =
    while (isRunning) {
      cnt += 1
      val valueArr = new Array[Float](128)
      (0 until 128).foreach(i => valueArr(i) = i)
      val input = ClusterServingInput(cnt.toString, valueArr)
      ctx.collect(List((cnt.toString, input)))
  }

  override def cancel(): Unit = {
  }
}
class MySink extends RichSinkFunction[List[(String, String)]] {
  override def invoke(value: List[(String, String)], context: SinkFunction.Context[_]): Unit = {
    value.foreach(kv => {
      println(s"Id ${kv._1} Cluster Serving inference result is ${ArrowDeserializer(kv._2)}")
    })
  }
}
