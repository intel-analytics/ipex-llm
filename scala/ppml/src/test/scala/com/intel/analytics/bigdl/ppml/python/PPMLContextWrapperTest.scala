package com.intel.analytics.bigdl.ppml.python

import com.intel.analytics.bigdl.ppml.PPMLContext
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.FunSuite

import java.util

class PPMLContextWrapperTest extends FunSuite {
  val ppmlContextWrapper: PPMLContextWrapper[Float] = PPMLContextWrapper.ofFloat

  def initArgs(): util.Map[String, String] = {
    val args = new util.HashMap[String, String]()
    args.put("kms_type", "SimpleKeyManagementService")
    args.put("simple_app_id", "465227134889")
    args.put("simple_app_key", "799072978028")
    args.put("primary_key_path", this.getClass.getClassLoader.getResource("primaryKey").getPath)
    args.put("data_key_path", this.getClass.getClassLoader.getResource("dataKey").getPath)
    args
  }

  def initAndRead(cryptoMode: String, path: String): Unit = {
    val appName = "test"
    val args = initArgs()

    val sc = ppmlContextWrapper.createPPMLContext(appName, args)
    val encryptedDataFrameReader = ppmlContextWrapper.read(sc, cryptoMode)
    ppmlContextWrapper.option(encryptedDataFrameReader, "header", "true")
    val df = ppmlContextWrapper.csv(encryptedDataFrameReader, path)

    assert(df.count() == 100)
  }

  def initAndWrite(df: DataFrame, encryptMode: String): Unit = {
    val appName = "test"
    val args = initArgs()

    val sc = ppmlContextWrapper.createPPMLContext(appName, args)
    ppmlContextWrapper.write(sc, df, encryptMode)
      .mode("overwrite")
      .option("header", true)
      .csv(this.getClass.getClassLoader.getResource("out").getPath)
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
    val cryptoMode = "plain_text"
    val path = this.getClass.getClassLoader.getResource("people.csv").getPath

    initAndRead(cryptoMode, path)
  }

  test("read encrypted csv file") {
    val cryptoMode = "AES/CBC/PKCS5Padding"
    val path = this.getClass.getClassLoader.getResource("encrypt-people").getPath

    initAndRead(cryptoMode, path)
  }

  test(" write plain text csv file") {
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]").appName("testData")
      .getOrCreate()

    val data = Seq(("Java", "20000"), ("Python", "100000"), ("Scala", "3000"))

    val df = spark.createDataFrame(data).toDF("language", "user")

    initAndWrite(df, "plain_text")
  }

  test(" write encrypted csv file") {
    val spark: SparkSession = SparkSession.builder()
      .master("local[1]").appName("testData")
      .getOrCreate()

    val data = Seq(("Java", "20000"), ("Python", "100000"), ("Scala", "3000"))

    val df = spark.createDataFrame(data).toDF("language", "user")

    initAndWrite(df, "AES/CBC/PKCS5Padding")
  }

}
