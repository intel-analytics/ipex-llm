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

package com.intel.analytics.bigdl.ppml.kms

import java.util.Base64

import scala.collection.mutable
import scala.collection.mutable.HashMap
import scala.util.Random

import org.apache.hadoop.conf.Configuration
import com.azure.identity.{DefaultAzureCredential, DefaultAzureCredentialBuilder}
import com.azure.security.keyvault.keys.KeyClientBuilder
import com.azure.security.keyvault.keys.cryptography.{CryptographyClient, CryptographyClientBuilder}
import com.azure.security.keyvault.keys.models.KeyType
import com.azure.security.keyvault.keys.cryptography.models.WrapResult
import com.azure.security.keyvault.keys.cryptography.models.KeyWrapAlgorithm
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.utils.KeyReaderWriter

class AzureKeyManagementService(keyVaultName: String, managedIdentityClientId : String = "")
  extends KeyManagementService {
  private val keyReaderWriter = new KeyReaderWriter
  private val cryptoClientMap: HashMap[String, CryptographyClient] =
    new mutable.HashMap[String, CryptographyClient]()

  private var defaultCredential: DefaultAzureCredential = null

  // support user managed identity
  if (managedIdentityClientId != null && managedIdentityClientId != "") {
    defaultCredential = new DefaultAzureCredentialBuilder()
      .managedIdentityClientId(managedIdentityClientId)
      .build()
  } else {
    defaultCredential = new DefaultAzureCredentialBuilder()
      .build()
  }
  private val keyClient = new KeyClientBuilder()
    .vaultUrl(s"https://${keyVaultName}.vault.azure.net/")
    .credential(defaultCredential)
    .buildClient()

  def retrievePrimaryKey(primaryKeySavePath: String = "", config: Configuration = null): Unit = {
    Log4Error.invalidInputError(primaryKeySavePath != null && primaryKeySavePath != "",
      "primaryKeySavePath should be specified")
    val keyName = "key-" + (1 to 4).map(x => Random.nextInt(10)).mkString
    val primaryKey = keyClient.createKey(keyName, KeyType.RSA)
    val keyId = primaryKey.getId()
    keyReaderWriter.writeKeyToFile(primaryKeySavePath, keyId, config)
  }

  def retrieveDataKey(primaryKeyPath: String, dataKeySavePath: String,
                      config: Configuration = null): Unit = {
    Log4Error.invalidInputError(primaryKeyPath != null && primaryKeyPath != "",
      "primaryKeyPath should be specified")
    Log4Error.invalidInputError(dataKeySavePath != null && dataKeySavePath != "",
      "dataKeySavePath should be specified")
    val primaryKeyId: String = keyReaderWriter.readKeyFromFile(primaryKeyPath, config)
    // get crypto client for primary key
    val cryptoClient = getCryptoClient(primaryKeyId)
    // create aes data key
    val aesKey = new Array[Byte](32)
    Random.nextBytes(aesKey)
    val keyString = Base64.getEncoder.encodeToString(aesKey)
    // wrap data key content.
    val wrapResult: WrapResult = cryptoClient.wrapKey(KeyWrapAlgorithm.RSA_OAEP, aesKey)
    val dataKeyCiphertext = Base64.getEncoder.encodeToString(wrapResult.getEncryptedKey())
    keyReaderWriter.writeKeyToFile(dataKeySavePath, dataKeyCiphertext, config)
  }

  def retrieveDataKeyPlainText(primaryKeyPath: String, dataKeyPath: String,
                               config: Configuration = null): String = {
    Log4Error.invalidInputError(primaryKeyPath != null && primaryKeyPath != "",
      "primaryKeyPath should be specified")
    Log4Error.invalidInputError(dataKeyPath != null && dataKeyPath != "",
      "dataKeyPath should be specified")
    val primaryKeyId: String = keyReaderWriter.readKeyFromFile(primaryKeyPath, config)
    val cryptoClient = getCryptoClient(primaryKeyId)
    val dataKeyCiphertext: String = keyReaderWriter.readKeyFromFile(dataKeyPath, config)
    val unwrapResult = cryptoClient.unwrapKey(KeyWrapAlgorithm.RSA_OAEP,
      Base64.getDecoder().decode(dataKeyCiphertext))
    val dataKey = unwrapResult.getKey()
    val dataKeyPlaintext: String = Base64.getEncoder.encodeToString(dataKey)
    dataKeyPlaintext
  }

  private def getCryptoClient(keyId: String): CryptographyClient = {
    if (cryptoClientMap.contains(keyId)) {
      cryptoClientMap(keyId)
    } else {
      val cryptoClient = new CryptographyClientBuilder()
        .credential(defaultCredential)
        .keyIdentifier(keyId)
        .buildClient()
      cryptoClientMap + (keyId -> cryptoClient)
      cryptoClient
    }
  }
}

