/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.common

import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.{SparkConf, SparkContext}

/**
 * [[NNContext]] wraps a spark context in Analytics Zoo.
 *
 */
object NNContext {

  /**
   * Gets a SparkContext with optimized configuration for BigDL performance. The method
   * will also initialize the BigDL engine.

   * Note: if you use spark-shell or Jupyter notebook, as the Spark context is created
   * before your code, you have to set Spark conf values through command line options
   * or properties file, and init BigDL engine manually.
   *
   * @param conf User defined Spark conf
   * @return Spark Context
   */
  def getNNContext(conf: SparkConf = null): SparkContext = {
    val bigdlConf = Engine.createSparkConf(conf)
    val sc = SparkContext.getOrCreate(bigdlConf)
    Engine.init
    sc
  }
}

