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
import java.nio.file.{Files, Paths}

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, FileSystem, Path}
import org.apache.hadoop.io.IOUtils

object File {
  private[bigdl] val hdfsPrefix: String = "hdfs:"

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
      var os: FileOutputStream = null
      var objFile: ObjectOutputStream = null
      try {
        os = new FileOutputStream(fileName)
        objFile = new ObjectOutputStream(os)
        objFile.writeObject(obj)
        objFile.close()
        os.close()
      } finally {
        if (null != objFile) objFile.close()
        if (null != os) os.close()
      }
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
    var fs: FileSystem = null
    var out: FSDataOutputStream = null
    var objFile: ObjectOutputStream = null
    try {
      fs = dest.getFileSystem(new Configuration())
      if (fs.exists(dest)) {
        if (overwrite) {
          fs.delete(dest, true)
        } else {
          throw new RuntimeException(s"file $fileName already exists")
        }
      }
      out = fs.create(dest)
      val byteArrayOut = new ByteArrayOutputStream()
      objFile = new ObjectOutputStream(byteArrayOut)
      objFile.writeObject(obj)
      IOUtils.copyBytes(new ByteArrayInputStream(byteArrayOut.toByteArray), out, 1024, true)
      objFile.close()
      out.close()
    } finally {
      if (null != objFile) objFile.close()
      if (null != out) out.close()
      if (null != fs) fs.close()
    }
  }

  /**
   * Load file from HDFS
   *
   * @param fileName
   */
  def loadFromHdfs[T](fileName: String): T = {
    val byteArrayOut = readHdfsByte(fileName)
    var objFile: ObjectInputStream = null
    try {
      objFile = new ObjectInputStream(new ByteArrayInputStream(byteArrayOut))
      val result = objFile.readObject()
      objFile.close()
      result.asInstanceOf[T]
    } finally {
      if (null != objFile) objFile.close()
    }
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
      var is: FileInputStream = null
      var objFile: ObjectInputStream = null
      try {
        is = new FileInputStream(fileName)
        objFile = new ObjectInputStream(is)
        val result = objFile.readObject().asInstanceOf[T]
        objFile.close()
        is.close()
        result
      } finally {
        if (null != objFile) objFile.close()
        if (null != is) is.close()
      }
    }
    res.asInstanceOf[T]
  }

  /**
   * load binary file from HDFS
   * @param fileName
   * @return
   */
  def readHdfsByte(fileName: String): Array[Byte] = {
    val src: Path = new Path(fileName)
    var fs: FileSystem = null
    var in: FSDataInputStream = null
    try {
      fs = src.getFileSystem(new Configuration())
      in = fs.open(src)
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      in.close()
      fs.close()
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
    }
  }
}
