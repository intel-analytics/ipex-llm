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

package com.intel.analytics.bigdl.ppml.fl

import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.log4j.LogManager
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
 * VflContext is a singleton object holding a FLClient object
 * For multiple vfl usage of an application, only one FLClient exists thus avoiding Channel cost
 */
object FLContext {
  val logger = LogManager.getLogger(getClass)
  var flClient: FLClient = null
  var sparkSession: SparkSession = null

  def resetFLContext(): Unit = {
    flClient = null
  }

  def setPsiSalt(psiSalt: String): Unit = {
    flClient.psiSalt = psiSalt
  }

  def getPsiSalt(): String = {
      flClient.psiSalt
  }

  def initFLContext(id: Int, target: String = null): Unit = {
    createSparkSession()
    Engine.init

    if (flClient == null) {
      this.synchronized {
        if (flClient == null) {
          flClient = new FLClient()
          flClient.setClientId(id)
          if (target != null) {
            flClient.setTarget(target)
          }
          flClient.build()
          logger.info(s"Created FlClient with ID: ${flClient.getClientUUID}, target: $target")
        }
      }
    }

  }

  def getClient(): FLClient = {
    flClient
  }

  def getSparkSession(): SparkSession = {
    if (sparkSession == null) {
      createSparkSession()
    }
    sparkSession
  }

  def createSparkSession(): Unit = {
    this.synchronized {
      if (sparkSession == null) {
        val conf = new SparkConf().setMaster("local[*]")
        sparkSession = SparkSession
          .builder()
          .config(conf)
          .getOrCreate()
      }
    }
  }
}
