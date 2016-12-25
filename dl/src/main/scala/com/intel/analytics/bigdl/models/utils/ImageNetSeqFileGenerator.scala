/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.bigdl.models.utils

import java.nio.file.{Files, Paths}

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.image.{BGRImgToLocalSeqFile, LocalImgReader, LocalImageFiles}
import scopt.OptionParser

object ImageNetSeqFileGenerator {

  case class ImageNetSeqFileGeneratorParams(
    folder: String = ".",
    output: String = ".",
    parallel: Int = 1,
    blockSize: Int = 12800,
    train: Boolean = true,
    validate: Boolean = true
  )

  private val parser = new OptionParser[ImageNetSeqFileGeneratorParams]("Spark-DL ImageNet " +
    "Sequence File Generator") {
    head("Spark-DL ImageNet Sequence File Generator")
    opt[String]('f', "folder")
      .text("where you put the ImageNet data")
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
    opt[Unit]('t', "trainOnly")
      .text("only generate train data")
      .action((_, c) => c.copy(validate = false))
    opt[Unit]('v', "validationOnly")
      .text("only generate validation data")
      .action((_, c) => c.copy(train = false))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new ImageNetSeqFileGeneratorParams()).map(param => {
      if (param.train) {
        // Process train data
        println("Process train data...")
        val trainFolderPath = Paths.get(param.folder, "train")
        require(Files.isDirectory(trainFolderPath),
          s"${trainFolderPath} is not valid")
        val trainDataSet = DataSet.ImageFolder.paths(trainFolderPath)
        trainDataSet.shuffle()
        (0 until param.parallel).map(tid => {
          val workingThread = new Thread(new Runnable {
            override def run(): Unit = {
              val pipeline = trainDataSet -> LocalImgReader(256) ->
                BGRImgToLocalSeqFile(param.blockSize, Paths.get(param.output, "train",
                  s"imagenet-seq-$tid"))
              val iter = pipeline.toLocal().data(train = false)
              while (iter.hasNext) {
                println(s"Generated file ${iter.next()}")
              }
            }
          })
          workingThread.setDaemon(false)
          workingThread.start()
          workingThread
        }).foreach(_.join())
      }

      if (param.validate) {
        // Process validation data
        println("Process validation data...")
        val validationFolderPath = Paths.get(param.folder, "val")
        require(Files.isDirectory(validationFolderPath),
          s"${validationFolderPath} is not valid")

        val validationDataSet = DataSet.ImageFolder.paths(validationFolderPath)
        validationDataSet.shuffle()
        val iter = validationDataSet.data(train = false)
        (0 until param.parallel).map(tid => {
          val workingThread = new Thread(new Runnable {
            override def run(): Unit = {
              val imageIter = LocalImgReader(256)(iter)
              val fileIter = BGRImgToLocalSeqFile(param.blockSize, Paths.get(param.output, "val",
                  s"imagenet-seq-$tid"))(imageIter)
              while (fileIter.hasNext) {
                println(s"Generated file ${fileIter.next()}")
              }
            }
          })
          workingThread.setDaemon(false)
          workingThread.start()
          workingThread
        }).foreach(_.join())
      }
    })

    println("Done")
  }
}
