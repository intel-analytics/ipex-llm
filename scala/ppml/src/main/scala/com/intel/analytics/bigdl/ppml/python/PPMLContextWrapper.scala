package com.intel.analytics.bigdl.ppml.python

import com.intel.analytics.bigdl.ppml.PPMLContext
import org.slf4j.{Logger, LoggerFactory}

object PPMLContextWrapper {
  def ofFloat: PPMLContextWrapper[Float] = new PPMLContextWrapper[Float]()

  def ofDouble: PPMLContextWrapper[Double] = new PPMLContextWrapper[Double]()
}

class PPMLContextWrapper[T]() {
  val logger: Logger = LoggerFactory.getLogger(getClass)

  def createPPMLContext(): PPMLContext = {
    logger.info("create PPMLContextWrapper")
    PPMLContext.initPPMLContext("testApp")
  }

  def loadKeys(sc: PPMLContext,
               primaryKeyPath: String, dataKeyPath: String): Unit = {
    sc.loadKeys(primaryKeyPath, dataKeyPath)
  }
}
