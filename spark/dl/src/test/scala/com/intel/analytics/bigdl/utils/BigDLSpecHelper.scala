/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.utils

import java.io.{File => JFile}

import org.apache.log4j.Logger
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.mutable.ArrayBuffer

abstract class BigDLSpecHelper extends FlatSpec with Matchers with BeforeAndAfter {
  protected val logger = Logger.getLogger(getClass)

  private val tmpFiles : ArrayBuffer[JFile] = new ArrayBuffer[JFile]()

  protected def createTmpFile(): JFile = {
    val file = java.io.File.createTempFile("UnitTest", "BigDLSpecBase")
    logger.info(s"created file $file")
    tmpFiles.append(file)
    file
  }

  protected def getFileFolder(path: String): String = {
    path.substring(0, path.lastIndexOf(JFile.separator))
  }

  protected def getFileName(path: String): String = {
    path.substring(path.lastIndexOf(JFile.separator) + 1)
  }

  def doAfter(): Unit = {}

  def doBefore(): Unit = {}

  before {
    doBefore()
  }

  after {
    doAfter()
    tmpFiles.foreach(f => {
      if (f.exists()) {
        require(f.isFile, "cannot clean folder")
        f.delete()
        logger.info(s"deleted file $f")
      }
    })
  }
}
