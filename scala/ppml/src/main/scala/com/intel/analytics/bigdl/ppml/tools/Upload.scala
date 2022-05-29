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
package com.intel.analytics.bigdl.ppml.tools

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{Crypto, CryptoMode, EncryptRuntimeException, FernetEncrypt}
import com.intel.analytics.bigdl.ppml.utils.EncryptIOArguments
import com.intel.analytics.bigdl.dllib.utils.{File, Log4Error}
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, SimpleKeyManagementService}
import org.apache.hadoop.fs.{LocatedFileStatus, Path, RemoteIterator}
import org.slf4j.LoggerFactory

import java.io.{BufferedReader, BufferedWriter, DataInputStream, InputStreamReader, OutputStreamWriter}
import java.util.concurrent.locks.{Lock, ReentrantLock}
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.reflect.ClassTag

object Upload {
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

    val kms = arguments.kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        new EHSMKeyManagementService(arguments.kmsServerIP, arguments.kmsServerPort,
          arguments.ehsmAPPID, arguments.ehsmAPPKEY)
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        SimpleKeyManagementService(arguments.simpleAPPID, arguments.simpleAPPKEY)
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }

    val thread = 10
    val outCryptos = Array.tabulate(thread)(_ => Crypto(arguments.outputEncryptMode))
    val dataKeyPlaintext = kms.retrieveDataKeyPlainText(
      arguments.primaryKeyPath, arguments.dataKeyPath)
    val bufferedLines = new Array[String](thread)
    val hdfsBlockSize = 256 * 1024 * 1024
    val lineDelimiter = ","
    val rowDelimiter = "|"

    // val inputFiles = new File(arguments.inputPath).listFiles
    inputFiles.foreach { file => {
      val filePrefix: String = file.getPath.getName()
      if (fs.exists(new Path(arguments.outputPath)) == false) {
        fs.mkdirs(new Path(arguments.outputPath))
      }
      val splitNum = arguments.outputPartitionNum
      val inputStream = fs.open(file.getPath)
      val outputStream = fs.create(new Path(arguments.outputPath, filePrefix), true)
      val bufferedReader = new BufferedReader(new InputStreamReader(inputStream), 1024 * 1024)
      var readLineNum = read(bufferedReader, bufferedLines)
      while(readLineNum > 0) {
        val encyptedBytes = (0 until readLineNum).par.map { i =>
          val fernetCryptos = outCryptos(i)
          val line = bufferedLines(i)
          fernetCryptos.encryptBytes(line.getBytes, dataKeyPlaintext)
        }.toArray
        encyptedBytes.foreach { eb =>
          outputStream.writeInt(eb.length)
          outputStream.write(eb)
        }
        readLineNum = read(bufferedReader, bufferedLines)
      }
      inputStream.close()
      bufferedReader.close()
      outputStream.flush()
      outputStream.close()

      val c = Crypto(arguments.outputEncryptMode)
      val t = fs.open(new Path(arguments.outputPath, filePrefix))
      (0 to 10).foreach{ i =>
        val l = t.readInt()
        val buffer = new Array[Byte](l)
        print(l)
        t.read(buffer, 0, l)
        val db = c.decryptBytes(buffer, dataKeyPlaintext)
        println(new String(db))
      }
    }}

  }

  def toArray[E: ClassTag](iter: RemoteIterator[E]): Array[E] = {
    val buffer = new ArrayBuffer[E]()
    while(iter.hasNext) {
      buffer.append(iter.next())
    }
    buffer.toArray
  }

  def read(br: BufferedReader, buffer: Array[String]): Int = {
    var i = 0
    var line = br.readLine()
    while(line != null) {
      buffer(i) = line
      i += 1
      if (i < buffer.length) {
        line = br.readLine()
      } else {
        line = null
      }
    }
    i
  }
}
