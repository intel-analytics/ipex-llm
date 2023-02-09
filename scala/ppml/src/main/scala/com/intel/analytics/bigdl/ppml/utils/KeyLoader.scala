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

// load both encryptedDataKey and dataKeyPlainText
case class KeyLoader(val fromKms: Boolean,
                     val primaryKeyMaterial: String = "",
                     val kms: KeyManagementService = null,
                     val primaryKeyPlainText: String = "") extends Serializable {
    protected val keySize = 32
    
    // get an existing data key
    def getDataKeyPlainText(fileDirPath: String): String = {
        val metaPath = new Path(fileDirPath + "/.meta").toString
        val encryptedDataKey = new KeyReaderWriter.readKeyFromFile(metaPath)
        if (fromKms) {
            kms.retrieveDataKeyPlainText(primaryKeyMaterial, encryptedDataKey)
        } else {
            val dataKeyPlainTextBytes = new BigDLEncrypt()
                .init(AES_CBC_PKCS5PADDING, DECRYPT, primaryKeyPlainText)
                .doFinal(encryptedDataKey.getBytes)._1
            new String(dataKeyPlainTextBytes)
        }
    }

    // generate a data key and write it to meta as well
    def generateDataKeyPlainText(fileDirPath: String): String = {
        val metaPath = new Path(fileDirPath + "/.meta").toString
        if(fromKms) {
            kms.retrieveDataKey(primaryKeyMaterial, metaPath)
            kms.retrieveDataKeyPlainText(primaryKeyMaterial, metaPath)
        } else {
            val generator = KeyGenerator.getInstance("AES")
            generator.init(keySize, SecureRandom.getInstanceStrong())
            val key: SecretKey = generator.generateKey()
            val dataKeyPlainText = Base64.getEncoder().encodeToString(key.getEncoded())
            val dataKeyCiphertextBytes = new BigDLEncrypt()
                .init(AES_CBC_PKCS5PADDING, ENCRYPT, primaryKeyPlainText)
                .doFinal(dataKeyPlainText.getBytes)._1
            new KeyReaderWriter.writeKeyToFile(metaPath, new String(dataKeyPlainTextBytes))
            dataKeyPlainText
        }
    }
}

class KeyLoaderManagement extends Serializable {
    // map from primaryKeyName to KeyLoader
    var multiKeyLoaders = new HashMap[String, KeyLoader]
    
    def setKeyLoader(primaryKeyName: String, keyLoader: KeyLoader): Unit = {
        Log4Error.invalidInputError(!(multiKeyLoaders.contains(primaryKeyName)),
                                    s"keyLoaders with name $name are replicated.")
        multiKeyLoaders += (primaryKeyName -> keyLoader)
    }
    
    def getKeyLoader(primaryKeyName: String): KeyLoader = {
        Log4Error.invalidInputError(multiKms.contains(name),
                                    s"cannot get a not-existing kms.")
        multiKeyLoaders.get(primaryKeyName).get
    }

}
