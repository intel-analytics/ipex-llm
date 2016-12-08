package org.apache.spark.sparkExtension

import java.util.UUID

import org.apache.spark.storage.{BlockId, TempLocalBlockId, TestBlockId}

object SparkExtension {
def getLocalBlockId(id: String): BlockId = {
  new TestBlockId(id)
}
}


