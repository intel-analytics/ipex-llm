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

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.flink.streaming.api.functions.sink.{RichSinkFunction, SinkFunction}
import org.apache.flink.streaming.api.functions.source.{RichParallelSourceFunction, SourceFunction}
import org.apache.flink.streaming.api.scala.{StreamExecutionEnvironment, _}
import org.apache.log4j.{Level, Logger}

object ClusterServingInferenceBlockOthersExample {
  class MySource extends RichParallelSourceFunction[String] {
    var cnt = 0
    @volatile var isRunning = true

    override def open(parameters: Configuration): Unit = {
      println("Source opened")
    }
    override def run(ctx: SourceFunction.SourceContext[String]): Unit =
      while (isRunning) {
        cnt += 1
        ctx.collect(cnt.toString)
      }

    override def cancel(): Unit = {
    }
  }
  class MyMap extends RichMapFunction[String, String] {
    override def map(value: String): String = {
      println(s"preprocessing")
      Model.synchronized {
        while (Model.queueing != 0) {
          println("waiting during preprocessing")
          Model.wait()
        }
      }
      Thread.sleep(50)
      println(s"predicting")
      var res = ""
      var holder: String = null
      Model.synchronized {
        Model.queueing += 1
        while (Model.model == null) {
          println("waiting during predict")
          Model.wait()
        }
        holder = Model.model
        Model.model = null
      }
      println(s"Model taken, name is: $holder")
      res = holder + ": " + value
      Thread.sleep(200)
      Model.synchronized {
        Model.model = "MyModel"
        Model.queueing -= 1
        Model.notifyAll()
      }
      res
    }
  }
  class MySink extends RichSinkFunction[String] {
    override def invoke(value: String, context: SinkFunction.Context[_]): Unit = {
      println(s"value $value written to sink.")
    }
  }
  Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]): Unit = {
    val serving = StreamExecutionEnvironment.getExecutionEnvironment
    serving.addSource(new MySource())
      .map(new MyMap())
      .addSink(new MySink())
    serving.execute()
  }
}

object Model {
  var queueing: Int = 0
  var model: String = "MyModel"
}
