/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.common

import java.io._
import java.nio.file.attribute.PosixFilePermissions
import java.nio.file.{Path => JPath}

import org.apache.commons.io.filefilter.WildcardFileFilter
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, FileSystem, Path}
import org.apache.hadoop.io.IOUtils
import org.apache.log4j.Logger

import scala.collection.mutable.ArrayBuffer

private[zoo] object Utils {

  private val logger = Logger.getLogger(getClass)

  def listLocalFiles(path: String): Array[File] = {
    val files = new ArrayBuffer[File]()
    listFiles(path, files)
    files.toArray
  }

  /**
   * List files in local file system
   * @param path String
   * @param files File handles will be appended to files
   */
  def listFiles(path: String, files: ArrayBuffer[File]): Unit = {
    val file = new File(path)
    if (file.isDirectory) {
      file.listFiles().foreach(x => listFiles(x.getAbsolutePath, files))
    } else if (file.isFile) {
      files.append(file)
    } else {
      val filter = new WildcardFileFilter(file.getName)
      file.getParentFile.listFiles(new FilenameFilter {
        override def accept(dir: File, name: String): Boolean = filter.accept(dir, name)
      }).foreach(x => listFiles(x.getAbsolutePath, files))
    }
  }

  /**
   * List paths of local or remote files (HDFS, S3 and FTP etc)
   * with FileSystem API
   * @param path String path
   * @param recursive Recursive or not
   * @return Array[String]
   */
  def listPaths(path: String, recursive: Boolean = false): Array[String] = {
    val fs = getFileSystem(path)
    // List remote or local files
    val res = new ArrayBuffer[String]()
    try {
      val files = fs.listFiles(new Path(path), recursive)
      while (files.hasNext) {
        val file = files.next()
        // Ignore dir
        if (!file.isDirectory) {
          // Add file paths (string) into ArrayBuffer
          res.append(file.getPath.toString)
        }
      }
    } catch {
      case _: FileNotFoundException => logger.warn(s"$path doesn't exist!")
      case _: IOException => logger.error(s"List paths of $path error!")
    }
    fs.close()
    res.toArray
  }

  /**
   * Read all bytes of file (local or remote) and return bytes Array.
   * WARNING: Don't use it to read large files. It may cause performance issue
   * and OOM.
   * @param path String
   * @return Array[Byte]
   */
  def readBytes(path: String): Array[Byte] = {
    var fs: FileSystem = null
    var in: FSDataInputStream = null
    try {
      fs = getFileSystem(path)
      in = fs.open(new Path(path))
      val byteArrayOut = new ByteArrayOutputStream()
      IOUtils.copyBytes(in, byteArrayOut, 1024, true)
      byteArrayOut.toByteArray
    } finally {
      if (null != in) in.close()
      if (null != fs) fs.close()
    }
  }

  /**
   * Read all bytes of multiple files (local or remote) and
   * return 2 dim bytes Array.
   * WARNING: Don't use it to read large files. It may cause performance issue
   * and OOM.
   * @param paths String paths in Array
   * @return 2 dim Byte Array
   */
  def readBytes(paths: Array[String]): Array[Array[Byte]] = {
    paths.map(readBytes)
  }

  /**
   * Write string lines into given path (local or remote file system)
   * @param path String path
   * @param lines String content
   */
  def writeLines(path: String, lines: String): Unit = {
    val fs = getFileSystem(path)
    val outStream = fs.create(new Path(path), true)
    try {
      outStream.writeBytes(lines)
    } finally {
      outStream.close()
    }
    fs.close()
  }

  /**
   * Get FileSystem (local or remote) from given file path
   * @param fileName file path (string)
   * @param newInstance always new instance if is set to true, otherwise will
   *                    get from cache (may shared with our connections)
   * @return hadoop.fs.FileSystem
   */
  def getFileSystem(fileName: String, newInstance: Boolean = true): FileSystem = {
    if (newInstance) {
      FileSystem.newInstance(new Path(fileName).toUri, new Configuration())
    } else {
      FileSystem.get(new Path(fileName).toUri, new Configuration())
    }
  }

  /**
   * Get FileSystem (local or remote) from given file path
   * @param fileName file path (hadoop.fs.Path)
   * @param newInstance always new instance if is set to true, otherwise will
   *                    get from cache (may shared with our connections)
   * @return hadoop.fs.FileSystem
   */
  def getFileSystem(fileName: Path, newInstance: Boolean): FileSystem = {
    if (newInstance) {
      FileSystem.newInstance(fileName.toUri, new Configuration())
    } else {
      FileSystem.get(fileName.toUri, new Configuration())
    }
  }

  /**
   * Create file in FileSystem (local or remote)
   * @param path String path
   * @param overwrite overwrite exiting file or not
   * @return
   */
  def create(path: String, overwrite: Boolean = false): DataOutputStream = {
    getFileSystem(path).create(new Path(path), overwrite)
  }

  /**
   * Open file in FileSystem (local or remote)
   * @param path String path
   * @return DataInputStream
   */
  def open(path: String): DataInputStream = {
    getFileSystem(path).open(new Path(path))
  }

  /**
   * Save bytes into given path (local or remote file system).
   * WARNING: Don't use it to read large files. It may cause performance issue
   * and OOM.
   * @param bytes bytes
   * @param fileName String path
   * @param isOverwrite Overwrite exiting file or not
   */
  def saveBytes(bytes: Array[Byte], fileName: String, isOverwrite: Boolean = false): Unit = {
    var fs: FileSystem = null
    var out: FSDataOutputStream = null
    try {
      fs = getFileSystem(fileName)
      out = fs.create(new Path(fileName), isOverwrite)
      IOUtils.copyBytes(new ByteArrayInputStream(bytes), out, 1024, true)
    } finally {
      if (null != out) out.close()
      if (null != fs) fs.close()
    }
  }

  def logUsageErrorAndThrowException(errMessage: String, cause: Throwable = null): Unit = {
    logger.error(s"********************************Usage Error****************************\n"
      + errMessage)
    throw new AnalyticsZooException(errMessage, cause)
  }

  def createTmpDir(prefix: String = "Zoo", permissions: String = "rwx------"): JPath = {
    java.nio.file.Files.createTempDirectory(prefix,
      PosixFilePermissions.asFileAttribute(PosixFilePermissions.fromString(permissions)))
  }
}

class AnalyticsZooException(message: String, cause: Throwable)
  extends Exception(message, cause) {

  def this(message: String) = this(message, null)
}

