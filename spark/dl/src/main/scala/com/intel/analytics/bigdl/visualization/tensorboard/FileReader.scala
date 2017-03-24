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

import java.io.{BufferedInputStream, File, FileInputStream}
import java.nio.ByteBuffer

import org.tensorflow.util.Event

import scala.collection.mutable.ArrayBuffer
import scala.util.matching.Regex

object FileReader {
  val fileNameRegex = """bigdl.tfevents.*""".r

  /**
   * Search file with regex.
   * @param f
   * @param r
   * @return
   */
  private def recursiveListFiles(f: File, r: Regex): Array[File] = {
    val these = f.listFiles()
    val good = these.filter(f => r.findFirstIn(f.getName).isDefined)
    good ++ these.filter(_.isDirectory).flatMap(recursiveListFiles(_, r))
  }

  /**
   * List all events file in path.
   * @param path should be a folder.
   * @return
   */
  def listFiles(path: String): Array[File] = {
    val dir = new java.io.File(path)
    require(dir.isDirectory, s"FileReader: $path should be a directory")
    FileReader.recursiveListFiles(dir, fileNameRegex)
  }

  /**
   * List all folders contains event files in path.
   * @param path should be a folder.
   * @return
   */
  def list(path: String): Array[String] = {
    val dir = new java.io.File(path)
    require(dir.isDirectory, s"FileReader: $path should be a directory")
    FileReader.recursiveListFiles(dir, fileNameRegex).map(_.getParent).distinct
  }

  /**
   * Read all scalar events named tag from a path.
   * @param path should be a folder.
   * @param tag tag name.
   * @return
   */
  def readScalar(path: String, tag: String): Array[(Long, Float, Double)] = {
    val dir = new File(path)
    require(dir.isDirectory, s"FileReader: $path should be a directory")
    val files = dir.listFiles().filter(f => fileNameRegex.findFirstIn(f.getName()).isDefined)
    files.map{file =>
      readScalar(file, tag)
    }.flatMap(_.toIterator).sortWith(_._1 < _._1)
  }

  /**
   * Read all scalar events named tag from a file.
   * @param file
   * @param tag
   * @return
   */
  def readScalar(file: File, tag: String): Array[(Long, Float, Double)] = {
    val bis = new BufferedInputStream(new FileInputStream(file))
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

