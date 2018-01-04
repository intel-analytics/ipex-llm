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
import java.net.URI

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, FileSystem, Path}
import org.apache.hadoop.io.IOUtils

object File {
  private[bigdl] val hdfsPrefix: String = "hdfs:"
  private[bigdl] val s3aPrefix: String = "s3a:"

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
   * Save scala object into a local/hdfs/s3 path
   *
   * Notice: S3 path should be like s3a://bucket/xxx.
   *
   * See (hadoop aws)[http://hadoop.apache.org/docs/r2.7.3/hadoop-aws/tools/hadoop-aws/index.html]
   * for details, if you want to save model to s3.
   *
   * @param obj object to be saved.
   * @param fileName local/hdfs output path.
   * @param isOverwrite if overwrite.
   */
  def save(obj: Serializable, fileName: String, isOverwrite: Boolean = false): Unit = {
    var fw: FileWriter = null
    var out: OutputStream = null
    var objFile: ObjectOutputStream = null
    try {
      fw = FileWriter(fileName)
      out = fw.create(isOverwrite)
      objFile = new ObjectOutputStream(new BufferedOutputStream(out))
      objFile.writeObject(obj)
    } finally {
      if (null != objFile) objFile.close()
      if (null != out) out.close()
      if (null != fw) fw.close()
    }
  }

  def saveBytes(bytes: Array[Byte], fileName: String, isOverwrite: Boolean = false) : Unit = {
    var fw: FileWriter = null
    var out: OutputStream = null
    var objFile: ObjectOutputStream = null
    try {
      fw = FileWriter(fileName)
      out = fw.create(isOverwrite)
      IOUtils.copyBytes(new ByteArrayInputStream(bytes), out, 1024, true)
    } finally {
      if (null != objFile) objFile.close()
      if (null != out) out.close()
      if (null != fw) fw.close()
    }
  }

  private[bigdl] def getFileSystem(fileName: String): org.apache.hadoop.fs.FileSystem = {
    val src = new Path(fileName)
    val fs = src.getFileSystem(File.getConfiguration(fileName))
    require(fs.exists(src), src + " does not exists")
    fs
  }

  private[bigdl] def getConfiguration(fileName: String): Configuration = {
    if (fileName.startsWith(File.hdfsPrefix) || fileName.startsWith(s3aPrefix)) {
      new Configuration()
    } else {
      new Configuration(false)
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
   * Load a scala object from a local/hdfs/s3 path.
   *
   * Notice: S3 path should be like s3a://bucket/xxx.
   *
   * See (hadoop aws)[http://hadoop.apache.org/docs/r2.7.3/hadoop-aws/tools/hadoop-aws/index.html]
   * for details, if you want to load model from s3.
   *
   * @param fileName file name.
   */
  def load[T](fileName: String): T = {
    var fr: FileReader = null
    var in: InputStream = null
    var objFile: ObjectInputStream = null
    try {
      fr = FileReader(fileName)
      in = fr.open()
      val bis = new BufferedInputStream(in)
      val objFile = new ObjectInputStream(bis)
      objFile.readObject().asInstanceOf[T]
    } finally {
      if (null != in) in.close()
      if (null != fr) fr.close()
      if (null != objFile) objFile.close()
    }
  }

  def readBytes[T](fileName : String) : Array[Byte] = {
    var fr: FileReader = null
    var in: InputStream = null
    var objFile: ObjectInputStream = null
    try {
      fr = FileReader(fileName)
      in = fr.open()
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fr) fr.close()
      if (null != objFile) objFile.close()
    }
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
      fs = FileSystem.newInstance(new URI(fileName), new Configuration())
      in = fs.open(src)
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
    }
  }
}

/**
 * FileReader in BigDL.
 * @param fileName
 */
private[bigdl] class FileReader(fileName: String) {
  private var inputStream: InputStream = null
  private val conf = File.getConfiguration(fileName)
  private val path = new Path(fileName)
  private val fs: FileSystem = path.getFileSystem(conf)

  /**
   * get an InputStream
   * @return
   */
  def open(): InputStream = {
    require(inputStream == null, s"File $fileName has been opened already.")
    require(fs.exists(path), s"$fileName is empty!")
    inputStream = fs.open(path)
    inputStream
  }

  /**
   * close the resources.
   */
  def close(): Unit = {
    if (null != inputStream) inputStream.close()
    fs.close()
  }
}

object FileReader {
  private[bigdl] def apply(fileName: String): FileReader = {
    new FileReader(fileName)
  }
}

/**
 * FileWriter in BigDL.
 * @param fileName
 */
private[bigdl] class FileWriter(fileName: String) {
  private var outputStream: OutputStream = null
  private val conf = File.getConfiguration(fileName)
  private val path = new Path(fileName)
  private val fs: FileSystem = path.getFileSystem(conf)

  /**
   * get an OutputStream
   * @param overwrite if overwrite
   * @return
   */
  def create(overwrite: Boolean = false): OutputStream = {
    require(outputStream == null, s"File $fileName has been created already.")
    if (!overwrite) {
      require(!fs.exists(path), s"$fileName already exists!")
    }
    outputStream = fs.create(path, overwrite)
    outputStream
  }

  /**
   * close the resources.
   */
  def close(): Unit = {
    if (null != outputStream) outputStream.close()
    fs.close()
  }
}

object FileWriter {
  private[bigdl] def apply(fileName: String): FileWriter = {
    new FileWriter(fileName)
  }
}

