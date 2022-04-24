package com.intel.analytics.bigdl.ppml.fl.utils

import org.apache.logging.log4j.LogManager

trait TimingSupportive {
  val logger = LogManager.getLogger(getClass)

  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }
}
