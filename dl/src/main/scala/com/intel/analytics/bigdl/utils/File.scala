/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.utils

import java.io._
import java.nio.file._

import org.apache.avro.mapred.tether.OutputProtocol
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.Logger

object File {
  private[bigdl]  val localfsPrefix: String = "file:"
  private[bigdl]  val hdfsPrefix: String = "hdfs:"
  private[bigdl]  val log = Logger.getLogger(getClass)

  /**
   * Load torch object from a torch binary file
   *
   * @param fileName file name.
   * @return An instance of T
   */
  def loadTorch[T](fileName: String): T = {
    TorchFile.load[T](fileName)
  }

  /**
   * Save scala object into a torch binary file
   *
   * @param source  The object to be saved.
   * @param fileName file name to saving.
   * @param objectType The object type.
   * @param overWrite If over write.
   */
  def saveTorch(
      source: Any,
      fileName: String,
      objectType: TorchObject,
      overWrite: Boolean = false): Unit = {
    TorchFile.save(source, fileName, objectType, overWrite)
  }

  /**
   * Save scala object into a local/hdfs path
   *
   * @param obj object to be saved.
   * @param outputPath local/hdfs output path.
   * @param isOverwrite if overwrite.
   */
  def save(obj: Serializable, outputPath: String, isOverwrite: Boolean = false): Unit = {
    var localFilePath: String = outputPath
    if (outputPath.startsWith(File.localfsPrefix)) {
      localFilePath = outputPath.substring(File.localfsPrefix.length)
      if (Files.exists(Paths.get(outputPath)) && !isOverwrite) {
        throw new RuntimeException(s"file ${outputPath} already exists!")
      }
    } else {
      localFilePath = System.getProperty("user.dir") + "/output-model.tmp"
    }

    val objFile = new ObjectOutputStream(new FileOutputStream(localFilePath))
    objFile.writeObject(obj)
    objFile.close()

    // upload the model file to HDFS, overwrites if dest path already exists
    if (!outputPath.startsWith(File.localfsPrefix)) {
      moveFile2HDFS(localFilePath, outputPath, isOverwrite)
    }
  }

  /**
   * Move file from local to HDFS.
   *
   * @param localFilePath
   * @param hdfsPath
   * @param overwrite
   */
  def moveFile2HDFS(localFilePath: String, hdfsPath: String, overwrite: Boolean): Unit = {
    require(hdfsPath.startsWith(File.hdfsPrefix), s"path ${hdfsPath} should have prefix 'hdfs:'")
    val dest: Path = new Path(hdfsPath)
    val fs: FileSystem = dest.getFileSystem(new Configuration())
    if (fs.exists(dest)) {
      if (overwrite) {
        fs.delete(dest, true)
      } else {
        throw new RuntimeException(s"file $hdfsPath already exists")
      }
    }
    fs.moveFromLocalFile(new Path("file://" + localFilePath), dest)
  }

  /**
   * Copy file from HDFS to local.
   *
   * @param hdfsPath
   * @param localFilePath
   */
  def copyFile2Local(hdfsPath: String, localFilePath: String): Unit = {
    val src: Path = new Path(hdfsPath)
    val fs: FileSystem = src.getFileSystem(new Configuration())
    val dest: Path = new Path("file://" + localFilePath)
    fs.copyToLocalFile(false, src, dest, true)
  }

  /**
   * Load a scala object from a local/hdfs path.
   *
   * @param fileName file name.
   */
  def load[T](fileName: String): T = {
    var localFileName = fileName
    if (fileName.startsWith(File.hdfsPrefix)) {
      localFileName = System.getProperty("user.dir") + "/" + "model.tmp"
      copyFile2Local(fileName, localFileName)
    }
    val objFile = new ObjectInputStream(new FileInputStream(localFileName))
    try {
      objFile.readObject().asInstanceOf[T]
    } finally {
      objFile.close()
      if (fileName.startsWith(File.hdfsPrefix)) new File(localFileName).delete()
    }
  }
}
