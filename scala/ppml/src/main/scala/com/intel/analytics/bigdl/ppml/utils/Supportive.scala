package com.intel.analytics.bigdl.ppml.utils

import org.slf4j.LoggerFactory

trait Supportive {
  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    Supportive.logger.info(s"$name time elapsed $cost ms.")
    result
  }
}

object Supportive {
  val logger = LoggerFactory.getLogger(getClass)
}