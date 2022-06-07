
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
package com.intel.analytics.bigdl.ppml.examples.tpch

import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, ENCRYPT, EncryptRuntimeException, PLAIN_TEXT}
import com.intel.analytics.bigdl.ppml.utils.EncryptIOArguments
import com.intel.analytics.bigdl.dllib.utils.{File, Log4Error}
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, SimpleKeyManagementService}
import org.apache.hadoop.fs.{FSDataOutputStream, Path, RemoteIterator}
import org.slf4j.LoggerFactory
import java.io.{BufferedReader, InputStreamReader}
import java.util.concurrent.locks.{Lock, ReentrantLock}

import com.intel.analytics.bigdl.ppml.PPMLContext

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object SplitAndEncrypt {
  def main(args: Array[String]): Unit = {

    val logger = LoggerFactory.getLogger(getClass)
    val arguments = EncryptIOArguments.parser.parse(args, EncryptIOArguments()) match {
      case Some(arguments) => logger.info(s"starting with $arguments"); arguments
      case None => EncryptIOArguments.parser.failure("miss args, please see the usage info"); null
    }

    val outputEncryptMode = arguments.outputEncryptMode
    val fs = File.getFileSystem(arguments.inputPath)

    val inputPath = new Path(arguments.inputPath)
    Log4Error.invalidInputError(inputPath.isAbsolute, "inputPath should be absolute path")
    val inputFiles = toArray(fs.listFiles(inputPath, false))

    val conf = new SparkConf()
    val sc = PPMLContext.initPPMLContext(conf, "Split and encrypt")

    inputFiles.foreach { file => {
      val filePrefix: String = file.getPath.getName()
      if (fs.exists(new Path(arguments.outputPath)) == false) {
        fs.mkdirs(new Path(arguments.outputPath))
      }
      val df = sc.read(PLAIN_TEXT)
        .option("sep", "|")
        .csv(file.getPath)
        .repartition(arguments.outputPartitionNum)
      sc.write(df, AES_CBC_PKCS5PADDING).csv(new Path(arguments.outputPath, filePrefix))
    }
    }
  }

  def toArray[E: ClassTag](iter: RemoteIterator[E]): Array[E] = {
    val buffer = new ArrayBuffer[E]()
    while (iter.hasNext) {
      buffer.append(iter.next())
    }
    buffer.toArray
  }
}
