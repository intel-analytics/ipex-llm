package com.intel.analytics.bigdl.ppml

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
 * VflContext is a singleton object holding a FLClient object
 * For multiple vfl usage of an application, only one FLClient exists thus avoiding Channel cost
 */
object FLContext {
  var flClient: FLClient = null
  var sparkSession: SparkSession = null
  def initFLContext(target: String = null) = {
    this.synchronized {
      if (flClient == null) {
        this.synchronized {
          flClient = new FLClient()
          if (target != null) {
            flClient.setTarget(target)
          }
          flClient.build()
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
