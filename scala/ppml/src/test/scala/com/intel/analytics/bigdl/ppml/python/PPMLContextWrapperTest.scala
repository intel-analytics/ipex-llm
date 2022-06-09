package com.intel.analytics.bigdl.ppml.python

import com.intel.analytics.bigdl.ppml.PPMLContext
import org.scalatest.FunSuite

import java.util

class PPMLContextWrapperTest extends FunSuite {
  val ppmlContextWrapper: PPMLContextWrapper[Float] = PPMLContextWrapper.ofFloat

  def initArgs(): util.Map[String, String] = {
    val args = new util.HashMap[String, String]()
    args.put("spark.bigdl.kms.type", "SimpleKeyManagementService")
    args.put("spark.bigdl.kms.simple.id", "465227134889")
    args.put("spark.bigdl.kms.simple.key", "799072978028")
    args
  }

  test("init PPMLContext with app name") {
    val appName = "test"
    ppmlContextWrapper.createPPMLContext(appName)
  }

  test("init PPMLContext with app name & args") {
    val appName = "test"
    val args = initArgs()
    ppmlContextWrapper.createPPMLContext(appName, args)
  }

  test("read plain text csv file") {
    val appName = "test"
    val args = initArgs()
    val cryptoMode = "plain_text"
    val path = this.getClass.getClassLoader.getResource("people.csv").getPath
    
    val sc = ppmlContextWrapper.createPPMLContext(appName, args)
    val encryptedDataFrameReader = ppmlContextWrapper.read(sc, cryptoMode)
    ppmlContextWrapper.option(encryptedDataFrameReader, "header", "true")
    val df = ppmlContextWrapper.csv(encryptedDataFrameReader, path)

    assert(df.count() == 100)
  }

}
