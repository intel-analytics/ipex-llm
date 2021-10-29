package com.intel.analytics.bigdl.ppml.vfl.algorithm

import java.util

import com.intel.analytics.bigdl.ppml.vfl.VflContext

class PSI() {
  val flClient = VflContext.getClient()
  def getSalt(name: String, clientNum: Int, secureCode: String): String =
    flClient.psiStub.getSalt(name, clientNum, secureCode)

  def uploadSet(hashedIdArray: util.List[String]): Unit = {
    flClient.psiStub.uploadSet(hashedIdArray)
  }

  def downloadIntersection(): util.List[String] = flClient.psiStub.downloadIntersection
}
