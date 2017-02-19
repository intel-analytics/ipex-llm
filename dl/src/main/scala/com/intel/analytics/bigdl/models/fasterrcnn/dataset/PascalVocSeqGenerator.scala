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

package com.intel.analytics.bigdl.models.fasterrcnn.dataset

import java.io.File
import java.nio.file.Paths

import com.intel.analytics.bigdl.models.fasterrcnn.dataset.transformers.RoidbToSeqFile
import scopt.OptionParser

object PascalVocSeqGenerator {

  case class PascalVocSeqGeneratorParams(
    folder: String = ".",
    output: String = ".",
    parallel: Int = 1,
    blockSize: Int = 12800,
    imageSet: Option[String] = None
  )

  private val parser = new OptionParser[PascalVocSeqGeneratorParams]("Spark-DL Pascal VOC " +
    "Sequence File Generator") {
    head("Spark-DL Pascal VOC Sequence File Generator")
    opt[String]('f', "folder")
      .text("where you put the Pascol Voc data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('o', "output folder")
      .text("where you put the generated seq files")
      .action((x, c) => c.copy(output = x))
    opt[Int]('p', "parallel")
      .text("parallel num")
      .action((x, c) => c.copy(parallel = x))
    opt[Int]('b', "blockSize")
      .text("block size")
      .action((x, c) => c.copy(blockSize = x))
    opt[String]('i', "imageSet")
      .text("image set")
      .action((x, c) => c.copy(imageSet = Some(x)))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new PascalVocSeqGeneratorParams()).map(param => {
      val roidbs = if (param.imageSet.isDefined) {
        ObjectDataSource(param.imageSet.get, param.folder,
          useFlipped = false).roidbs
      } else {
        new File(param.folder).listFiles().map(f => Roidb(f.getAbsolutePath))
      }

      val total = roidbs.length
      val iter = roidbs.toIterator

      (0 until param.parallel).map(tid => {
        val workingThread = new Thread(new Runnable {
          override def run(): Unit = {
            val fileIter = RoidbToSeqFile(param.blockSize, Paths.get(param.output,
              s"$total-voc-seq-$tid"))(iter)
            while (fileIter.hasNext) {
              println(s"Generated file ${ fileIter.next() }")
            }
          }
        })
        workingThread.setDaemon(false)
        workingThread.start()
        workingThread
      }).foreach(_.join())
    })
  }
}
