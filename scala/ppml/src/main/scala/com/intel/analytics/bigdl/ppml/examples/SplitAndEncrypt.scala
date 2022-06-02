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
package com.intel.analytics.bigdl.ppml.examples

import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, ENCRYPT, EncryptRuntimeException, PLAIN_TEXT}
import com.intel.analytics.bigdl.ppml.utils.EncryptIOArguments
import com.intel.analytics.bigdl.dllib.utils.{File, Log4Error}
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, SimpleKeyManagementService}
import org.apache.hadoop.fs.{Path, RemoteIterator, FSDataOutputStream}
import org.slf4j.LoggerFactory

import java.io.{BufferedReader, InputStreamReader}
import java.util.concurrent.locks.{Lock, ReentrantLock}
import scala.collection.mutable.ArrayBuffer
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

    val dataKeyPlaintext = kms.retrieveDataKeyPlainText(arguments.primaryKeyPath, arguments.dataKeyPath)

    inputFiles.foreach{ file => {
      val begin = System.currentTimeMillis
      val thread = arguments.outputPartitionNum
      val outCryptos: Array[BigDLEncrypt] = Array.tabulate(thread)(_ => new BigDLEncrypt())
      (0 until thread).par.map { i =>
        outCryptos(i).init(arguments.outputEncryptMode, ENCRYPT, dataKeyPlaintext)
      }
      val bufferedLinesFirst = new Array[String](thread)
      val bufferedLinesSecond = new Array[String](thread)

      val filePrefix: String = file.getPath.getName()
      if (fs.exists(new Path(arguments.outputPath)) == false) {
        fs.mkdirs(new Path(arguments.outputPath))
      }

      val inputStream = fs.open(file.getPath)
      val outputStreamArray = new Array[FSDataOutputStream](thread)
      (0 until thread).par.map { i =>
        outputStreamArray(i) =  fs.create(Path.mergePaths(
          new Path(arguments.outputPath), new Path("/" + filePrefix + "/split_" + i.toString + ".csv")), true)
      }
      val bufferedReader = new BufferedReader(new InputStreamReader(inputStream), 1024 * 1024)

      // write header
      val header =  bufferedReader.readLine() + "\n"
      if(outputEncryptMode == AES_CBC_PKCS5PADDING) {
        (0 until thread).par.map { i =>
          val encryptedBytes = outCryptos(i).update(header.getBytes)
          outputStreamArray(i).write(encryptedBytes)
        }
      } else {
        (0 until thread).par.map { i =>
          outputStreamArray(i).write(header.getBytes)
        }
      }

      var readLineNum = read(bufferedReader, bufferedLinesFirst)
      readLineNum = read(bufferedReader, bufferedLinesSecond)
      while(readLineNum == thread) {
        (0 until readLineNum).par.map { i =>
          outputEncryptMode match {
            case AES_CBC_PKCS5PADDING => {
              val encryptedBytes = outCryptos(i).update(bufferedLinesFirst(i).getBytes)
              outputStreamArray(i).write(encryptedBytes)
            }
            case PLAIN_TEXT => {
              val outputStream = outputStreamArray(i)
              outputStream.write(bufferedLinesFirst(i).getBytes)
            }
            case default => {
              throw new EncryptRuntimeException("No such crypto mode!")
            }
          }
          bufferedLinesFirst(i) = bufferedLinesSecond(i)
        }
        readLineNum = read(bufferedReader, bufferedLinesSecond)
      }
      // last part
      if(outputEncryptMode == AES_CBC_PKCS5PADDING) {
        (0 until readLineNum).par.map { i =>
          val encryptedBytes = outCryptos(i).update(bufferedLinesFirst(i).getBytes)
          outputStreamArray(i).write(encryptedBytes)

          val (finalBytes, hmac) = outCryptos(i).doFinal(bufferedLinesSecond(i).getBytes)
          outputStreamArray(i).write(finalBytes)
          outputStreamArray(i).write(hmac)
        }
        (readLineNum until thread).par.map { i =>
          val (finalBytes, hmac) = outCryptos(i).doFinal(bufferedLinesFirst(i).getBytes)
          outputStreamArray(i).write(finalBytes)
          outputStreamArray(i).write(hmac)
        }
      } else {
        (0 until readLineNum).par.map { i =>
          outputStreamArray(i).write(bufferedLinesFirst(i).getBytes)
          outputStreamArray(i).write(bufferedLinesSecond(i).getBytes)
        }
        (readLineNum until thread).par.map { i =>
          outputStreamArray(i).write(bufferedLinesSecond(i).getBytes)
        }
      }
      (0 until thread).par.map { i =>
        outputStreamArray(i).flush()
        outputStreamArray(i).close()
      }
      inputStream.close()

      val end = System.currentTimeMillis
      val cost = (end - begin)
      println(s"Encrypt time elapsed $cost ms.")
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
        line = br.readLine() + "\n"
      } else {
        line = null
      }
    }
    i
  }
}
