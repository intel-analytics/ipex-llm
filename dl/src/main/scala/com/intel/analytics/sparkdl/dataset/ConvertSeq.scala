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
package com.intel.analytics.sparkdl.dataset

import java.io.IOException
import java.nio.ByteBuffer
import java.nio.file.Paths

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.{SequenceFile, Text}
import scopt.OptionParser

object ConvertSeq {

  case class ConvertSeqParams(
    folder: String = "./",
    outputSeq: String = "./",
    parallel: Int = 1,
    buffer : Int = 256,
    dataSetType: String = "ImageNet"
  )

  private val parser = new OptionParser[ConvertSeqParams]("Spark-DL Convert Seq") {
    head("Convert Image Files to Hadoop Sequential Files")
    opt[String]('f', "folder")
      .text("where you put the dataset")
      .action((x, c) => c.copy(folder = x))
    opt[String]('o', "outputSeq")
      .text("outputSeq folder")
      .action((x, c) => c.copy(outputSeq = x))
    opt[Int]('p', "parallel")
      .text("parallel num")
      .action((x, c) => c.copy(parallel = x))
    opt[Int]('b', "buffer")
      .text("buffer size")
      .action((x, c) => c.copy(buffer = x))
    opt[String]('d', "dataSetType")
      .text("dataset type")
      .action((x, c) => c.copy(dataSetType = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, new ConvertSeqParams()).map(param => {
      param.dataSetType match {
        case "ImageNet" =>
          val dataSource = new ImageNetDataSource(Paths.get(param.folder), looped = false)
          val pathToImage = PathToRGBImage(256)
          val worker = new Worker(dataSource -> pathToImage, param.parallel)
          worker.process(param.outputSeq)
        case "Cifar-10" =>
          val dataSource = new CifarDataSource(Paths.get(param.folder), looped = false)
          val arrayToImage = ArrayByteToRGBImage()
          val worker = new Worker(dataSource -> arrayToImage, param.parallel)
          worker.process(param.outputSeq)
        case _ => throw new UnsupportedOperationException(s"Only ImageNet/Cifar-10 supported")
      }
    })
  }
}

class Worker(dataSet: DataSource[RGBImage], parallel: Int) {

  def process(target: String): Unit = {
    var i = 0
    var file = s"${target}-seq"
    val writer = new Writer(file)
    while(dataSet.hasNext) {
      val data = dataSet.next()
      val imageKey = s"${data.label()}-${i}"
      println(s"write ${imageKey}")
      //writer.write(imageKey, RGBImage.convertToByte(data.content, data.width(), data.height()),
      //  data.width(), data.height())
      i += 1
    }
    writer.close()
  }
}

class Writer @throws[IOException]
(val seqFilePath: String) {
  private val conf: Configuration = new Configuration
  val path = new Path(seqFilePath)
  val writer = SequenceFile.createWriter(conf, SequenceFile.Writer.file(path),
    SequenceFile.Writer.keyClass(classOf[Text]), SequenceFile.Writer.valueClass(classOf[Text]))
  var preBuffer: ByteBuffer = ByteBuffer.allocate(4 * 2)

  @throws[Exception]
  def write(imageKey: String, img: Array[Byte], width: Int, height: Int) {
    preBuffer.putInt(width)
    preBuffer.putInt(height)
    val data: Array[Byte] = new Array[Byte](preBuffer.capacity + img.length)
    System.arraycopy(preBuffer.array, 0, data, 0, preBuffer.capacity)
    System.arraycopy(img, 0, data, preBuffer.capacity, img.length)
    preBuffer.clear
    writer.append(new Text(imageKey), new Text(data))
  }

  def close() {
    try {
      writer.close()
    } catch {
      case e: IOException =>
        e.printStackTrace()
    }
  }
}
