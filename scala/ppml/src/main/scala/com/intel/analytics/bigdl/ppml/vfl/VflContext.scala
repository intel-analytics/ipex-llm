package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.ppml.FLClient


/**
 * VflContext is a singleton object holding a FLClient object
 * For multiple vfl usage of an application, only one FLClient exists thus avoiding Channel cost
 */
object VflContext {
  var flClient: FLClient = null
  def initContext() = {
    this.synchronized {
      if (flClient == null) {
        this.synchronized {
          flClient = new FLClient()
        }
      }
    }
  }
  def getClient(): FLClient = {
    flClient
  }
}
