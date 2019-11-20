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

package com.intel.analytics.bigdl.models.utils

import com.intel.analytics.bigdl.dataset.segmentation.{COCODataset, COCOSerializeContext}
import java.io.File
import java.nio.file.{Files, Paths}
import java.util.concurrent.atomic.AtomicInteger
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.SequenceFile.Writer
import org.apache.hadoop.io.compress.BZip2Codec
import org.apache.hadoop.io.{BytesWritable, SequenceFile}
import scala.collection.parallel.ForkJoinTaskSupport
import scopt.OptionParser

object COCOSeqFileGenerator {

  /**
   * Configuration class for COCO sequence file
   * generator
   *
   * @param folder the COCO image files location
   * @param metaPath the metadata json file location
   * @param output generated seq files location
   * @param parallel number of parallel
   * @param blockSize block size
   */
  case class COCOSeqFileGeneratorParams(
    folder: String = ".",
    metaPath: String = "instances_val2014.json",
    output: String = ".",
    parallel: Int = 1,
    blockSize: Int = 12800
  )

  private val parser = new OptionParser[COCOSeqFileGeneratorParams]("BigDL COCO " +
    "Sequence File Generator") {
    head("BigDL COCO Sequence File Generator")
    opt[String]('f', "folder")
      .text("where you put the COCO image files")
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
    opt[String]('m', "metaPath")
      .text("metadata json file path")
      .action((x, c) => c.copy(metaPath = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, COCOSeqFileGeneratorParams()).foreach { param =>
      println("Loading COCO metadata")
      val meta = COCODataset.load(param.metaPath, param.folder)
      println("Metadata loaded")
      val conf: Configuration = new Configuration
      val doneCount = new AtomicInteger(0)
      val tasks = meta.images.filter(img => {
        val path = img.path
        val valid = Files.exists(path) && !Files.isDirectory(path)
        if (!valid) {
          System.err.print(s"[Warning] The image file ${path.getFileName} does not exist.\n")
        }
        valid
      }).grouped(param.blockSize).zipWithIndex.toArray.par
      tasks.tasksupport = new ForkJoinTaskSupport(
        new scala.concurrent.forkjoin.ForkJoinPool(param.parallel))
      tasks.foreach { case (imgs, blkId) =>
        val outFile = new Path(param.output, s"coco-seq-$blkId.seq")
        val key = new BytesWritable
        val value = new BytesWritable
        val writer = SequenceFile.createWriter(conf, Writer.file(outFile), Writer.keyClass(key
          .getClass), Writer.valueClass(value.getClass), Writer.compression(SequenceFile
          .CompressionType.BLOCK, new BZip2Codec))
        val context = new COCOSerializeContext
        imgs.foreach { img =>
          context.clear()
          context.dump(img.fileName)
          img.dumpTo(context)
          context.dump(COCODataset.MAGIC_NUM)
          val keyBytes = context.toByteArray
          key.set(keyBytes, 0, keyBytes.length)
          val bytes = img.data
          value.set(bytes, 0, bytes.length)
          writer.append(key, value)
          val cnt = doneCount.incrementAndGet()
          if (cnt % 500 == 0) {
            System.err.print(s"\r$cnt / ${meta.images.length} = ${cnt.toFloat/meta.images.length}")
          }
        }
        writer.close()
      }
      System.err.print("\n")
    }
  }
}
