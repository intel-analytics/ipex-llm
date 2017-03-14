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

import java.io._
import java.nio.file._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.IOUtils

object File {
  private[bigdl]  val hdfsPrefix: String = "hdfs:"

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
   * @param fileName local/hdfs output path.
   * @param isOverwrite if overwrite.
   */
  def save(obj: Serializable, fileName: String, isOverwrite: Boolean = false): Unit = {
    if (fileName.startsWith(File.hdfsPrefix)) {
      saveToHdfs(obj, fileName, isOverwrite)
    } else {
      if (Files.exists(Paths.get(fileName)) && !isOverwrite) {
        throw new RuntimeException("file exists!")
      }
      val objFile = new ObjectOutputStream(new FileOutputStream(fileName))
      objFile.writeObject(obj)
    }
  }

  /**
   * Write file to HDFS.
   * @param obj
   * @param fileName
   * @param overwrite
   */
  def saveToHdfs(obj: Serializable, fileName: String, overwrite: Boolean): Unit = {
    require(fileName.startsWith(File.hdfsPrefix),
      s"hdfs path ${fileName} should have prefix 'hdfs:'")
    val dest = new Path(fileName)
    val fs = dest.getFileSystem(new Configuration())
    if (fs.exists(dest)) {
      if (overwrite) {
        fs.delete(dest, true)
      } else {
        throw new RuntimeException(s"file $fileName already exists")
      }
    }
    val out = fs.create(dest)
    val byteArrayOut = new ByteArrayOutputStream()
    val objFile = new ObjectOutputStream(byteArrayOut)
    objFile.writeObject(obj)
    IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
  }

  /**
   * Load file from HDFS
   *
   * @param fileName
   * @param localFilePath
   */
  def loadFromHdfs[T](fileName: String, localFilePath: String = null): T = {
    val src: Path = new Path(fileName)
    val fs = src.getFileSystem(new Configuration())
    val in = fs.open(src)
    val byteArrayOut = new ByteArrayOutputStream()
    IOUtils.copyBytes(in, byteArrayOut, 1024, true)
    val objFile = new ObjectInputStream(new ByteArrayInputStream(byteArrayOut.toByteArray))
    objFile.readObject().asInstanceOf[T]
  }


  /**
   * Load a scala object from a local/hdfs path.
   *
   * @param fileName file name.
   */
  def load[T](fileName: String): T = {
    val res = if (fileName.startsWith(File.hdfsPrefix)) {
      loadFromHdfs[T](fileName)
    } else {
      val objFile = new ObjectInputStream(new FileInputStream(fileName))
      objFile.readObject().asInstanceOf[T]
    }
    res.asInstanceOf[T]
  }
}
