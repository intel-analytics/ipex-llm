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

package com.intel.analytics.bigdl.ppml.crypto

import com.intel.analytics.bigdl.dllib.utils.File
import com.intel.analytics.bigdl.ppml.crypto.dataframe.DataFrameHelper
import org.apache.hadoop.fs.Path

import java.io.FileWriter
import java.nio.file.{Files, Paths}
import scala.io.Source

class EncryptSpec extends DataFrameHelper {
  val (plainFileName, encryptFileName, data, dataKeyPlaintext) = generateCsvData()
  val fs = File.getFileSystem(plainFileName)

  "decrypt file" should "work" in {
    val decryptFile = dir + "/decrypt_file.csv"
    val crypto = new BigDLEncrypt()
    crypto.init(AES_CBC_PKCS5PADDING, DECRYPT, dataKeyPlaintext)
    crypto.doFinal(encryptFileName, decryptFile)
    val readData = Source.fromFile(decryptFile).getLines().mkString("\n")
    readData + "\n" should be (data)
  }

  "encrypt stream" should "work" in {
    val encrypt = new BigDLEncrypt()
    encrypt.init(AES_CBC_PKCS5PADDING, ENCRYPT, dataKeyPlaintext)
    val bis = fs.open(new Path(plainFileName))
    val outs = fs.create(new Path(dir + "/en_o.csv"))
    encrypt.doFinal(bis, outs)
    bis.close()
    outs.flush()
    outs.close()
    Thread.sleep(1000)

    val decrypt = new BigDLEncrypt()
    decrypt.init(AES_CBC_PKCS5PADDING, DECRYPT, dataKeyPlaintext)
    val bis2 = fs.open(new Path(dir + "/en_o.csv"))
    val outs2 = fs.create(new Path(dir + "/de_o.csv"))
    decrypt.doFinal(bis2, outs2)
    outs2.close()
    outs2.flush()
    bis2.close()
    val originFile = Files.readAllBytes(Paths.get(plainFileName))
    val deFile = Files.readAllBytes(Paths.get(dir.toString, "/de_o.csv"))
    originFile.sameElements(deFile) should be (true)
  }
}
