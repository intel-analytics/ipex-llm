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

package com.intel.analytics.bigdl.ppml.utils

import java.io.Serializable
import java.security.SecureRandom
import javax.crypto.{KeyGenerator, SecretKey}
import java.util.Base64
import scala.collection.mutable.HashMap
import com.intel.analytics.bigdl.ppml.kms.KeyManagementService
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, BigDLEncrypt, DECRYPT, ENCRYPT}
import com.intel.analytics.bigdl.ppml.utils.KeyReaderWriter
import org.apache.hadoop.fs.Path

// load both encryptedDataKey and dataKeyPlainText
case class KeyLoader(val fromKms: Boolean,
                     val primaryKeyMaterial: String = "",
                     val kms: KeyManagementService = null,
                     val primaryKeyPlainText: String = "") extends Serializable {
    protected val keySize = 32
    val keyReaderWriter = new KeyReaderWriter
    val META_FILE_NAME = ".meta"
    val CRYPTO_MODE = AES_CBC_PKCS5PADDING
    
    // retrieve an existing data key
    def retrieveDataKeyPlainText(fileDirPath: String): String = {
        val metaPath = new Path(fileDirPath + "/" + META_FILE_NAME).toString
        val encryptedDataKey = keyReaderWriter.readKeyFromFile(metaPath)
        if (fromKms) {
            kms.retrieveDataKeyPlainText(primaryKeyMaterial, encryptedDataKey)
        } else {
            val decrypt = new BigDLEncrypt()
            decrypt.init(CRYPTO_MODE, DECRYPT, primaryKeyPlainText)
            new String(decrypt.doFinal(encryptedDataKey.getBytes)._1)
        }
    }

    // generate a data key and write it to meta as well
    def generateDataKeyPlainText(fileDirPath: String): String = {
        val metaPath = new Path(fileDirPath + "/" + META_FILE_NAME).toString
        if(fromKms) {
            kms.retrieveDataKey(primaryKeyMaterial, metaPath)
            kms.retrieveDataKeyPlainText(primaryKeyMaterial, metaPath)
        } else {
            val generator = KeyGenerator.getInstance("AES")
            generator.init(keySize, SecureRandom.getInstanceStrong())
            val key: SecretKey = generator.generateKey()
            val dataKeyPlainText = Base64.getEncoder().encodeToString(key.getEncoded())
            val encrypt = new BigDLEncrypt()
            encrypt.init(CRYPTO_MODE, ENCRYPT, primaryKeyPlainText)
            val encryptedDataKey = new String(
              encrypt.doFinal(dataKeyPlainText.getBytes)._1
            )
            keyReaderWriter.writeKeyToFile(metaPath, encryptedDataKey)
            dataKeyPlainText
        }
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
                                    s"cannot get a not-existing kms.")
        multiKeyLoaders.get(primaryKeyName).get
    }

}
