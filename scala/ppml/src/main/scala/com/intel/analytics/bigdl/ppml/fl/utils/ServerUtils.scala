package com.intel.analytics.bigdl.ppml.fl.utils

import com.intel.analytics.bigdl.dllib.utils.Log4Error

object ServerUtils {
  def checkClientId(clientNum: Int, id: String): Boolean = {
    try {
      if (id.toInt <= 0 || id.toInt > clientNum) {
        throw new Exception("Invalid client ID")
      } else {
        true
      }
    } catch {
      case e: Exception => false
    }
  }
}
