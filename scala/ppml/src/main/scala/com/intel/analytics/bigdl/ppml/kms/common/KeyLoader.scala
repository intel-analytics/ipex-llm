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

package com.intel.analytics.bigdl.ppml.kms.common

import java.io.{Serializable, File, FileWriter}
import java.security.SecureRandom
import javax.crypto.{KeyGenerator, Cipher, SecretKey}
import javax.crypto.spec.SecretKeySpec
import java.util.Base64
import java.net.URI
import java.nio.charset.StandardCharsets
import scala.collection.mutable.HashMap
import com.intel.analytics.bigdl.ppml.kms.KeyManagementService
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, DECRYPT, ENCRYPT}
import com.intel.analytics.bigdl.ppml.utils.KeyReaderWriter
import com.intel.analytics.bigdl.ppml.kms.common.KmsMetaFormatSerializer
import org.apache.hadoop.fs.{Path, FileSystem}
import org.apache.hadoop.conf.Configuration
import org.apache.spark.sql.SparkSession

// load both encryptedDataKey and dataKeyPlainText
case class KeyLoader(val fromKms: Boolean,
                     val primaryKeyMaterial: String = "",
                     val kms: KeyManagementService = null,
                     val primaryKeyPlainText: String = "",
                     config: Configuration = null) extends Serializable {
    protected val keySize = 128
    protected val keyReaderWriter = new KeyReaderWriter
    val META_FILE_NAME = ".meta"
    protected val CRYPTO_MODE = AES_CBC_PKCS5PADDING
    protected var encryptedDataKey: String = ""
    protected val hadoopConfig = if (config != null) config else new Configuration()
    protected val encrypterType = SparkSession.builder().getOrCreate()
      .sparkContext.hadoopConfiguration
      .get("spark.bigdl.encryter.type", BigDLEncrypt.COMMON).toLowerCase

    // retrieve the plaintext string of an existing data key
    def retrieveDataKeyPlainText(fileDirPath: String): String = {
        val encryptedDataKey = encrypterType match {
          case BigDLEncrypt.NATIVE_AES_CBC =>
            // native AES CBC uses file granularity instead of folder
            val filePath = new Path(fileDirPath).toString
            val fs: FileSystem = FileSystem.get(new URI(filePath), hadoopConfig)
            val inStream = fs.open(new Path(filePath))
            val iv = new Array[Byte](16)
            inStream.read(iv)
            new String(iv, StandardCharsets.UTF_8)
          case BigDLEncrypt.COMMON =>
            val metaPath = new Path(fileDirPath + "/" + META_FILE_NAME).toString
            val fs: FileSystem = FileSystem.get(new URI(metaPath), hadoopConfig)
            val inStream = fs.open(new Path(metaPath))
            val jsonStr = scala.io.Source.fromInputStream(inStream)
              .takeWhile(_ != null).mkString
            KmsMetaFormatSerializer(jsonStr).encryptedDataKey
        }

        val dataKeyPlainText = if (fromKms) {
          kms.retrieveDataKeyPlainText(primaryKeyMaterial, "", null, encryptedDataKey)
        } else {
          encrypterType match {
            case BigDLEncrypt.NATIVE_AES_CBC => primaryKeyPlainText
            case BigDLEncrypt.COMMON =>
              // unwrap the data key with primaryKeyPlainText
              val cipher = dataKeyCipher(Cipher.DECRYPT_MODE)
              val encryptedDataKeyBytes = Base64.getDecoder().decode(encryptedDataKey)
              val dataKeyPlainTextBytes = cipher.doFinal(encryptedDataKeyBytes)
              new String(dataKeyPlainTextBytes, StandardCharsets.UTF_8)
          }
        }

        SparkSession.builder().getOrCreate().sparkContext.hadoopConfiguration
          .set(s"bigdl.read.dataKey.$encryptedDataKey.plainText", dataKeyPlainText)
        dataKeyPlainText
    }

    // generate a data key, and return the plaintext string
    // and cache encryptedDataKey for meta writing after df writing
    def generateDataKeyPlainText(): (String, String) = {
        if (fromKms) {
            encryptedDataKey = kms.retrieveDataKey(primaryKeyMaterial).get
            val dataKeyPlainText = kms.retrieveDataKeyPlainText(primaryKeyMaterial,
              "", null, encryptedDataKey)
            (dataKeyPlainText, encryptedDataKey)
        } else {
          encrypterType match {
            case BigDLEncrypt.NATIVE_AES_CBC => (primaryKeyPlainText, null)
            case BigDLEncrypt.COMMON =>
              val dataKeyPlainText: String = {
                val generator = KeyGenerator.getInstance("AES")
                generator.init(keySize, SecureRandom.getInstanceStrong())
                val key: SecretKey = generator.generateKey()
                Base64.getEncoder().encodeToString(key.getEncoded)
              }
              encryptedDataKey = {
                val cipher = dataKeyCipher(Cipher.ENCRYPT_MODE)
                val encryptedDataKeyBytes = cipher.doFinal(
                  dataKeyPlainText.getBytes(StandardCharsets.UTF_8))
                Base64.getEncoder().encodeToString(encryptedDataKeyBytes)
              }
              (dataKeyPlainText, encryptedDataKey)
          }
        }
    }

    // write encryptedDataKey to meta after spark df has been written
    def writeEncryptedDataKey(fileDirPath: String): Unit = {
      val metaPath = new Path(fileDirPath + "/" + META_FILE_NAME).toString
      val jsonStr = KmsMetaFormatSerializer(KmsMetaFormat(encryptedDataKey))
      keyReaderWriter.writeKeyToFile(metaPath, jsonStr, hadoopConfig)
    }

    def dataKeyCipher(operateMode: Int): Cipher = {
      val decodedPrimaryKey = Base64.getDecoder().decode(primaryKeyPlainText)
      val skp = new SecretKeySpec(decodedPrimaryKey, 0, decodedPrimaryKey.length, "AES")
      val cipher = Cipher.getInstance("AES")
      cipher.init(operateMode, skp)
      cipher
    }
}

class KeyLoaderManagement extends Serializable {
    // map from primaryKeyName to KeyLoader
    var multiKeyLoaders = new HashMap[String, KeyLoader]
    def addKeyLoader(primaryKeyName: String, keyLoader: KeyLoader): Unit = {
        Log4Error.invalidInputError(!(multiKeyLoaders.contains(primaryKeyName)),
                                    s"keyLoaders with name $primaryKeyName are replicated.")
        multiKeyLoaders += (primaryKeyName -> keyLoader)
    }
    def retrieveKeyLoader(primaryKeyName: String): KeyLoader = {
        Log4Error.invalidInputError(multiKeyLoaders.contains(primaryKeyName),
                                    s"cannot get a not existing keyLoader.")
        multiKeyLoaders.get(primaryKeyName).get
    }

    def count(): Int = multiKeyLoaders.size

}

