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

package com.intel.analytics.bigdl.visualization.tensorboard

import java.io.{BufferedInputStream}
import java.nio.ByteBuffer

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.tensorflow.util.Event

import scala.collection.mutable.ArrayBuffer
import scala.util.matching.Regex

private[bigdl] object FileReader {
  val fileNameRegex = """bigdl.tfevents.*""".r

  /**
   * Search file with regex.
   * @param f
   * @param r
   * @return
   */
  private def recursiveListFiles(f: Path, r: Regex, fs: FileSystem): Array[Path] = {
    val buffer = new ArrayBuffer[Path]()
    val files = fs.listFiles(f, true)
    while (files.hasNext) {
      val file = files.next().getPath
      if (r.findFirstIn(file.getName).isDefined) {
        buffer.append(file)
      }
    }
    buffer.toArray
  }

  /**
   * List all events file in path.
   * @param path should be a local/HDFS folder.
   * @return
   */
  def listFiles(path: String): Array[Path] = {
    val logPath = new Path(path)
    val fs = logPath.getFileSystem(new Configuration(false))
    require(fs.isDirectory(logPath), s"FileReader: $path should be a directory")
    FileReader.recursiveListFiles(logPath, fileNameRegex, fs)
  }

  /**
   * List all folders contains event files in path.
   * @param path should be a local/HDFS folder.
   * @return
   */
  def list(path: String): Array[String] = {
    val logPath = new Path(path)
    val fs = logPath.getFileSystem(new Configuration(false))
    require(fs.isDirectory(logPath), s"FileReader: $path should be a directory")
    FileReader.recursiveListFiles(logPath, fileNameRegex, fs).map(_.getParent.toString).distinct
  }

  /**
   * Read all scalar events named tag from a path.
   * @param path should be a local/HDFS folder.
   * @param tag tag name.
   * @return
   */
  def readScalar(path: String, tag: String): Array[(Long, Float, Double)] = {
    val logPath = new Path(path)
    val fs = logPath.getFileSystem(new Configuration(false))
    require(fs.isDirectory(logPath), s"FileReader: $path should be a directory")
    val files = FileReader.recursiveListFiles(logPath, fileNameRegex, fs)
    files.map{file =>
      readScalar(file, tag, fs)
    }.flatMap(_.toIterator).sortWith(_._1 < _._1)
  }

  /**
   * Read all scalar events named tag from a file.
   * @param file The path of file. Support local/HDFS path.
   * @param tag tag name.
   * @return
   */
  def readScalar(file: Path, tag: String, fs: FileSystem): Array[(Long, Float, Double)] = {
    require(fs.isFile(file), s"FileReader: ${file} should be a file")
    val bis = new BufferedInputStream(fs.open(file))
    val longBuffer = new Array[Byte](8)
    val crcBuffer = new Array[Byte](4)
    val bf = new ArrayBuffer[(Long, Float, Double)]
    while (bis.read(longBuffer) > 0) {
      val l = ByteBuffer.wrap(longBuffer.reverse).getLong()
      bis.read(crcBuffer)
      // TODO: checksum
      //      val crc1 = ByteBuffer.wrap(crcBuffer.reverse).getInt()
      val eventBuffer = new Array[Byte](l.toInt)
      bis.read(eventBuffer)
      val e = Event.parseFrom(eventBuffer)
      if (e.getSummary.getValueCount == 1 &&
        tag.equals(e.getSummary.getValue(0).getTag())) {
        bf.append((e.getStep, e.getSummary.getValue(0).getSimpleValue,
          e.getWallTime))
      }
      bis.read(crcBuffer)
      //      val crc2 = ByteBuffer.wrap(crcBuffer.reverse).getInt()
    }
    bis.close()
    bf.toArray.sortWith(_._1 < _._1)
  }
}

