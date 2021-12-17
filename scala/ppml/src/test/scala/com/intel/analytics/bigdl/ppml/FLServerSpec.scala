package com.intel.analytics.bigdl.ppml

import com.intel.analytics.bigdl.ppml.example.LogManager
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FLServerSpec extends FlatSpec with Matchers with BeforeAndAfter with LogManager {
  "start server from config" should "work" in {
    val flServer = new FLServer(Array("-c",
      getClass.getClassLoader.getResource("ppml-conf-2-party.yaml").getPath))
    flServer.build()
    flServer.start()
    val flClient = new FLClient(Array("-c",
      getClass.getClassLoader.getResource("ppml-conf-2-party.yaml").getPath))
    flClient.build()
  }

}
