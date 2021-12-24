package com.intel.analytics.bigdl.ppml.example

import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator


trait LogManager {
  Configurator.setLevel("org", Level.ERROR)
  Configurator.setLevel("com.intel.analytics.bigdl.ppml", Level.DEBUG)
}
