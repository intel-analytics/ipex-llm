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

import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, ENCRYPT, EncryptRuntimeException, BigDLEncrypt, PLAIN_TEXT}
import com.intel.analytics.bigdl.ppml.utils.EncryptIOArguments
import com.intel.analytics.bigdl.dllib.utils.{File, Log4Error}
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, SimpleKeyManagementService}
import org.apache.hadoop.fs.{Path, RemoteIterator}
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

    inputFiles.par.map{ file => {
      val filePrefix: String = file.getPath.getName()
      if (fs.exists(new Path(arguments.outputPath)) == false) {
        fs.mkdirs(new Path(arguments.outputPath))
      }
      val splitNum = arguments.outputPartitionNum
      var inputStream = fs.open(file.getPath)
      var bufferedReader = new BufferedReader(new InputStreamReader(inputStream), 1024 * 1024)
      val header =  bufferedReader.readLine()
      val numLines = bufferedReader.lines().count()

      inputStream.close()
      inputStream = fs.open(file.getPath)
      bufferedReader = new BufferedReader(new InputStreamReader(inputStream), 1024 * 1024)
      bufferedReader.readLine()

      val linesPerFile: Long = numLines / splitNum  // lines per split file
      val splitArray = new Array[Long](splitNum) // split-point
      for(i <- 0 to splitNum-1) {
        splitArray(i) = linesPerFile
      }
      splitArray(splitNum-1) += numLines % linesPerFile // for last part

      var currentSplitNum: Int = 0
      val rtl: Lock = new ReentrantLock()
      val begin = System.currentTimeMillis
      splitArray.par.map{
        num => {
          var splitContentString = ""
          var splitFileName = ""
          rtl.lock()
          try { // get split content
            splitContentString = header + "\n" + read(bufferedReader, num.toInt)
            splitFileName = filePrefix + "/split_" + currentSplitNum.toString + ".csv"
            currentSplitNum += 1
          } finally {
            rtl.unlock()
          }
          val outputStream = fs.create(Path.mergePaths(
            new Path(arguments.outputPath), new Path("/" + splitFileName)), true)

          outputEncryptMode match {
            case AES_CBC_PKCS5PADDING => {
              val crypto = new BigDLEncrypt()
              crypto.init(arguments.outputEncryptMode, ENCRYPT, dataKeyPlaintext)
              val encryptedBytes = crypto.doFinal(splitContentString.getBytes)
              val header = crypto.genHeader()
              outputStream.write(header)
              outputStream.write(encryptedBytes._1)
              outputStream.write(encryptedBytes._2)
              outputStream.flush()
              outputStream.close()
              println("Successfully save encrypted text " + filePrefix + "  " + splitFileName)
            }
            case PLAIN_TEXT => {
              outputStream.write(splitContentString.getBytes)
              outputStream.flush()
              outputStream.close()
              println("Successfully save plain text " + filePrefix + "  " + splitFileName)
            }
            case default => {
              throw new EncryptRuntimeException("No such crypto mode!")
            }
          }

        }
      }
      val end = System.currentTimeMillis
      val cost = (end - begin)
      println(s"Encrypt time elapsed $cost ms.")
      inputStream.close()
    }}

  }

  def toArray[E: ClassTag](iter: RemoteIterator[E]): Array[E] = {
    val buffer = new ArrayBuffer[E]()
    while(iter.hasNext) {
      buffer.append(iter.next())
    }
    buffer.toArray
  }

  def read(br: BufferedReader, numLines: Int): String = {
    var i = 0
    var result = ""
    while(i  < numLines) {
      i += 1
      result += (br.readLine() + "\n")
    }
    result
  }
}
