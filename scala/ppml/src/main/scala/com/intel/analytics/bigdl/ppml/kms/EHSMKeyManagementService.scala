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

import org.apache.hadoop.conf.Configuration
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.utils.HTTPUtil.postRequest
import com.intel.analytics.bigdl.ppml.utils.{EHSMParams, KeyReaderWriter}

object EHSM_CONVENTION {

  // Actions
  val ACTION_CREATE_KEY = "CreateKey"
  val ACTION_GENERATE_DATAKEY_WO_PLAINTEXT = "GenerateDataKeyWithoutPlaintext"
  val ACTION_DECRYPT = "Decrypt"

  // Request keys
  val PAYLOAD_KEYSPEC = "keyspec"
  val PAYLOAD_ORIGIN = "origin"
  val PAYLOAD_AAD = "aad"
  val PAYLOAD_KEY_ID = "keyid"
  val PAYLOAD_KEY_LENGTH = "keylen"
  val PAYLOAD_CIPHER_TEXT = "ciphertext"
  val PAYLOAD_PLAIN_TEXT = "plaintext"

  val KEYSPEC_EH_AES_GCM_128 = "EH_AES_GCM_128"

  val ORIGIN_EH_INTERNAL_KEY = "EH_INTERNAL_KEY"
}


class EHSMKeyManagementService(
      kmsServerIP: String,
      kmsServerPort: String,
      ehsmAPPID: String,
      ehsmAPIKEY: String)extends KeyManagementService {

  val keyReaderWriter = new KeyReaderWriter

  Log4Error.invalidInputError(ehsmAPPID != "", s"ehsmAPPID should not be empty string.")
  Log4Error.invalidInputError(ehsmAPIKEY != "", s"ehsmAPIKEY should not be empty string.")

  def retrievePrimaryKey(primaryKeySavePath: String, config: Configuration = null): Unit = {
    Log4Error.invalidInputError(primaryKeySavePath != null && primaryKeySavePath != "",
      "primaryKeySavePath should be specified")
    val action: String = EHSM_CONVENTION.ACTION_CREATE_KEY
    val currentTime = System.currentTimeMillis() // ms
    val timestamp = s"$currentTime"
    val ehsmParams = new EHSMParams(ehsmAPPID, ehsmAPIKEY, timestamp)
    ehsmParams.addPayloadElement(EHSM_CONVENTION.PAYLOAD_KEYSPEC,
      EHSM_CONVENTION.KEYSPEC_EH_AES_GCM_128)
    ehsmParams.addPayloadElement(EHSM_CONVENTION.PAYLOAD_ORIGIN,
      EHSM_CONVENTION.ORIGIN_EH_INTERNAL_KEY)

    val primaryKeyCiphertext: String = timing(
      "EHSMKeyManagementService request for primaryKeyCiphertext") {
      val postString: String = ehsmParams.getPostJSONString()
      val postResult = postRequest(constructUrl(action), postString)
      postResult.getString(EHSM_CONVENTION.PAYLOAD_KEY_ID)
    }
    keyReaderWriter.writeKeyToFile(primaryKeySavePath, primaryKeyCiphertext, config)
  }

  def retrieveDataKey(primaryKeyPath: String, dataKeySavePath: String,
                      config: Configuration = null): Unit = {
    Log4Error.invalidInputError(primaryKeyPath != null && primaryKeyPath != "",
      "primaryKeyPath should be specified")
    Log4Error.invalidInputError(dataKeySavePath != null && dataKeySavePath != "",
      "dataKeySavePath should be specified")
    val action = EHSM_CONVENTION.ACTION_GENERATE_DATAKEY_WO_PLAINTEXT
    val encryptedPrimaryKey: String = keyReaderWriter.readKeyFromFile(primaryKeyPath, config)
    val currentTime = System.currentTimeMillis() // ms
    val timestamp = s"$currentTime"
    val ehsmParams = new EHSMParams(ehsmAPPID, ehsmAPIKEY, timestamp)
    ehsmParams.addPayloadElement(EHSM_CONVENTION.PAYLOAD_AAD, "test")
    ehsmParams.addPayloadElement(EHSM_CONVENTION.PAYLOAD_KEY_ID, encryptedPrimaryKey)
    ehsmParams.addPayloadElement(EHSM_CONVENTION.PAYLOAD_KEY_LENGTH, "32")

    val dataKeyCiphertext: String = timing(
      "EHSMKeyManagementService request for dataKeyCiphertext") {
      val postString: String = ehsmParams.getPostJSONString()
      val postResult = postRequest(constructUrl(action), postString)
      postResult.getString(EHSM_CONVENTION.PAYLOAD_CIPHER_TEXT)
    }
    keyReaderWriter.writeKeyToFile(dataKeySavePath, dataKeyCiphertext, config)
  }


  override def retrieveDataKeyPlainText(primaryKeyPath: String, dataKeyPath: String,
                                        config: Configuration = null): String = {
    Log4Error.invalidInputError(primaryKeyPath != null && primaryKeyPath != "",
      "primaryKeyPath should be specified")
    Log4Error.invalidInputError(dataKeyPath != null && dataKeyPath != "",
      "dataKeyPath should be specified")
    val action: String = EHSM_CONVENTION.ACTION_DECRYPT
    val encryptedPrimaryKey: String = keyReaderWriter.readKeyFromFile(primaryKeyPath, config)
    val encryptedDataKey: String = keyReaderWriter.readKeyFromFile(dataKeyPath, config)
    val currentTime = System.currentTimeMillis() // ms
    val timestamp = s"$currentTime"
    val ehsmParams = new EHSMParams(ehsmAPPID, ehsmAPIKEY, timestamp)
    ehsmParams.addPayloadElement(EHSM_CONVENTION.PAYLOAD_AAD, "test")
    ehsmParams.addPayloadElement(EHSM_CONVENTION.PAYLOAD_CIPHER_TEXT, encryptedDataKey)
    ehsmParams.addPayloadElement(EHSM_CONVENTION.PAYLOAD_KEY_ID, encryptedPrimaryKey)
    val dataKeyPlaintext: String = timing(
      "EHSMKeyManagementService request for dataKeyPlaintext") {
      val postString: String = ehsmParams.getPostJSONString()
      val postResult = postRequest(constructUrl(action), postString)
      postResult.getString(EHSM_CONVENTION.PAYLOAD_PLAIN_TEXT)
    }
    dataKeyPlaintext
  }


  private def constructUrl(action: String): String = {
    s"http://$kmsServerIP:$kmsServerPort/ehsm?Action=$action"
  }
}
