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

package com.intel.analytics.bigdl.friesian.serving.utils

import com.codahale.metrics.Timer
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.logging.log4j.{LogManager, Logger}
import org.apache.spark.sql.SparkSession

object Utils {
  var helper: gRPCHelper = _
  val logger: Logger = LogManager.getLogger(getClass)

  def runMonitor(): Boolean = {
    if (helper.monitorPort == 0) {
      false
    } else {
      true
    }
  }

  def getPromBuckets: Array[Double] = {
    Array(0.001D, 0.005D, 0.008D, 0.01D, 0.015D, 0.020D, 0.025D, 0.03D, 0.04D, 0.05D, 0.06D,
      0.07D, 0.08D, 0.09D, 0.1D, 1D, 2D, 5D, 10D)
  }

  def loadUserData(dataDir: String, userIdCol: String, dataNum: Int): Array[Int] = {
    val spark = SparkSession.builder.getOrCreate
    val df = spark.read.parquet(dataDir)
    df.select(userIdCol).distinct.limit(dataNum).rdd.map(row => {
      //      row.getLong(0).toInt
      row.getInt(0)
    }).collect
  }

  /**
   * Use hadoop utils to copy file from remote to local
   * @param src remote path, could be hdfs, s3
   * @param dst local path
   */
  def copyToLocal(src: String, dst: String): Unit = {
    val conf = new Configuration()

    val srcPath = new Path(src)
    val fs = srcPath.getFileSystem(conf)

    val dstPath = new Path(dst)
    fs.copyToLocalFile(srcPath, dstPath)
  }
}

case class TimerMetrics(name: String,
                        count: Long,
                        meanRate: Double,
//                        min: Long,
//                        max: Long,
                        mean: Double,
                        median: Double,
//                        stdDev: Double,
                        _75thPercentile: Double,
                        _95thPercentile: Double,
//                        _98thPercentile: Double,
                        _99thPercentile: Double)
//                        _999thPercentile: Double)

object TimerMetrics {
  def apply(name: String, timer: Timer): TimerMetrics =
    TimerMetrics(
      name,
      timer.getCount,
      timer.getMeanRate,
//      timer.getSnapshot.getMin / 1000000,
//      timer.getSnapshot.getMax / 1000000,
      timer.getSnapshot.getMean / 1000000,
      timer.getSnapshot.getMedian / 1000000,
//      timer.getSnapshot.getStdDev / 1000000,
      timer.getSnapshot.get75thPercentile() / 1000000,
      timer.getSnapshot.get95thPercentile() / 1000000,
//      timer.getSnapshot.get98thPercentile() / 1000000,
      timer.getSnapshot.get99thPercentile() / 1000000
//      timer.getSnapshot.get999thPercentile() / 1000000
    )
}
