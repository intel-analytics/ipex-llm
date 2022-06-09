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

import com.intel.analytics.bigdl.ppml.utils.Supportive

object KMS_CONVENTION {
  val MODE_SIMPLE_KMS = "SimpleKeyManagementService"
  val MODE_EHSM_KMS = "EHSMKeyManagementService"
  val MODE_AZURE_KMS = "AzureKeyManagementService"
}


/**
 * KeyManagementService interface
 */
trait KeyManagementService extends Supportive {
  /**
   * Generate a primary key.
   * @param primaryKeySavePath the path to save primary key.
   */
  def retrievePrimaryKey(primaryKeySavePath: String)

  /**
   * Generate a data key and use primary key to encrypt it.
   * @param primaryKeyPath the path of primary key.
   * @param dataKeySavePath the path to save encrypted data key.
   */
  def retrieveDataKey(primaryKeyPath: String, dataKeySavePath: String)

  /**
   * Use primary key to decrypt data key.
   * @param primaryKeyPath the path of primary key.
   * @param dataKeyPath the path of encrypted data key.
   * @return the plaintext of data key.
   */
  def retrieveDataKeyPlainText(primaryKeyPath: String, dataKeyPath: String): String
}
