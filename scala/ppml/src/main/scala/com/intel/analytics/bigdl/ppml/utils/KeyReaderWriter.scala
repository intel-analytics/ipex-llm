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

package com.intel.analytics.bigdl.ppml.utils
import java.io.PrintWriter
import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.IOUtils

import scala.io.Source

class KeyReaderWriter {

  def writeKeyToFile(encryptedKeyPath: String, encryptedKeyContent: String,
                     config: Configuration = null): Unit = {
    val hadoopConfig = if (config != null) config else new Configuration()
    val fs: FileSystem = FileSystem.get(new URI(encryptedKeyPath), hadoopConfig)
    val outputStream = fs.create(new Path(encryptedKeyPath))
    outputStream.writeBytes(encryptedKeyContent + "\n")
    outputStream.close()
  }

  def readKeyFromFile(encryptedKeyPath: String, config: Configuration = null): String = {
    val hadoopConfig = if (config != null) config else new Configuration()
    val fs = FileSystem.get(new URI(encryptedKeyPath), hadoopConfig)
    val inStream = fs.open(new Path(encryptedKeyPath))
    val content = scala.io.Source.fromInputStream(inStream).getLines().next()
    content
  }

}

