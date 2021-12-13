package com.intel.analytics.bigdl.ppml.example

import org.apache.log4j.{Level, Logger}

trait LogManager {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.ppml").setLevel(Level.DEBUG)
}
