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
package com.intel.analytics.bigdl.utils

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/**
 * A trait which handles the creation of a [[SparkContext]] at the beginning
 * of the test suite and the finalization of its lifecyle when the test ends.
 */
trait SparkContextLifeCycle extends FlatSpec with BeforeAndAfter {
  var sc: SparkContext = null
  def nodeNumber: Int = 1
  def coreNumber: Int = 1
  def appName: String = "SparkApp"

  /**
   * Custom statements to execute inside [[before]] after a [[SparkContext]] is initialized.
   */
  def beforeTest: Any = {}
  /**
   * Custom statements to execute inside [[after]] after the [[SparkContext]] is stopped.
   */
  def afterTest: Any = {}

  before {
    Engine.init(nodeNumber, coreNumber, true)
    val conf = Engine.createSparkConf().setMaster(s"local[$coreNumber]").setAppName(appName)
    sc = SparkContext.getOrCreate(conf)
    beforeTest
  }

  after {
    if (sc != null) {
      sc.stop()
    }
    afterTest
  }
}
