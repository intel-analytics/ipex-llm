package com.intel.analytics.bigdl.friesian.nearline.utils

import org.apache.logging.log4j.{LogManager, Logger}

import java.io.File

object NearlineUtils {
  var helper: NearlineHelper = _
  val logger: Logger = LogManager.getLogger(getClass)

  def getListOfFiles(dir: String): Array[List[String]] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      logger.info("file exists & dir")
      val parquetList = d.listFiles.filter(_.isFile).toList.map(_.getAbsolutePath)
        .filter(path => !path.endsWith("SUCCESS") & !path.endsWith(".crc"))
      logger.info(s"ParquetList length: ${parquetList.length}")
      val batch = (parquetList.length.toFloat / helper.part).ceil.toInt
      parquetList.sliding(batch, batch).toArray
    } else {
      logger.info(s"empty, exists: ${d.exists()}, dir: ${d.isDirectory}")
      Array[List[String]]()
    }
  }
}
