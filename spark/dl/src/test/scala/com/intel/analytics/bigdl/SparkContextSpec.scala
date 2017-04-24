/*
 * Copyright 2017 The BigDL Authors.
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

package com.intel.analytics.bigdl

import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

/**
 * Superclass for Specs that need to set up and tear down a SparkContext
 */
private[bigdl] abstract class SparkContextSpec extends FlatSpec with Matchers with BeforeAndAfter {

  private var _sc: SparkContext = _

  // Tests can override these to change master, config
  private[bigdl] def getNodeNumber: Int = 1
  private[bigdl] def getCoreNumber: Int = 1
  private[bigdl] def getExtraConf: Map[String, String] = Map()
  
  private[bigdl] def sc: SparkContext = {
    if (_sc == null) {
      val conf = new SparkConf()
        .setAppName(getClass.getSimpleName)
        .setMaster(s"local[$getCoreNumber]")
        .set("spark.executor.cores", getCoreNumber.toString)
      conf.setAll(getExtraConf)
      val bigdlConf = Engine.createSparkConf(conf)
      _sc = SparkContext.getOrCreate(bigdlConf)
      Engine.init(bigdlConf)
      // hacky: always test in local mode but pretend like we have many nodes if > 1
      if (getNodeNumber > 1) {
        Engine.setNodeNumber(getNodeNumber)
      }
    }
    _sc
  }

  after {
    if (_sc != null) {
      _sc.stop()
      _sc = null
      Engine.reset()
    }
  }

}
