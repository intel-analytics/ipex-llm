package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.ppml.FLServer
import com.intel.analytics.bigdl.ppml.psi.test.TestUtils
import com.intel.analytics.bigdl.ppml.vfl.PSI
import scala.collection.JavaConverters._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class PSISpec extends FlatSpec with Matchers with BeforeAndAfter{
  "PSI get salt" should "work" in {
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    VflContext.initContext()
    val pSI = new PSI()
    val salt = pSI.getSalt()
    require(salt != null, "Get salt failed.")
  }
  "PSI upload set" should "work" in {
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    VflContext.initContext()
    val pSI = new PSI()
    val set = List("key1", "key2")
    val salt = pSI.getSalt()
    val hashedKey = TestUtils.parallelToSHAHexString(set.asJava, salt)
    pSI.uploadSet(hashedKey)

    require(salt != null, "Get salt failed.")
  }
  "PSI download intersection" should "work" in {
    val flServer = new FLServer()
    flServer.build()
    flServer.start()
    VflContext.initContext()
    val pSI1 = new PSI()
    val pSI2 = new PSI()
    val set1 = List("key1", "key2")
    val set2 = List("key2", "key3")
    val salt1 = pSI1.getSalt()
    val salt2 = pSI2.getSalt()
    val hashedKey1 = TestUtils.parallelToSHAHexString(set1.asJava, salt1)
    val hashedKey2 = TestUtils.parallelToSHAHexString(set2.asJava, salt2)
    pSI1.uploadSet(hashedKey1)
    pSI2.uploadSet(hashedKey2)
    val intersection = pSI1.downloadIntersection()
    require(intersection.size() == 1, "Intersection number is wrong.")
    require(intersection.get(0) == hashedKey1.get(1), "Intersection result is wrong.")
  }
}
