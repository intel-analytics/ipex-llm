package com.intel.analytics.bigdl.ppml.fl

import com.intel.analytics.bigdl.ppml.fl.example.DebugLogger
import com.intel.analytics.bigdl.ppml.fl.utils.PortUtils
import org.apache.logging.log4j.LogManager
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FLSpec extends FlatSpec with Matchers with BeforeAndAfter with DebugLogger{
  var port: Int = 8980
  var target: String = "localhost:8980"
  val logger = LogManager.getLogger(getClass)
  before {
    // try only next 3 ports, if failed, it may well be that server holds the port and fails to release
    port = PortUtils.findNextPortAvailable(port, port + 3)
    target = "localhost:" + port
    logger.info(s"Running test on port: $port, target: $target")

  }
}
