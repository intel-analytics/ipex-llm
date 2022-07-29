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
import org.apache.hadoop.conf.Configuration

object KMS_CONVENTION {
  val MODE_SIMPLE_KMS = "SimpleKeyManagementService"
  val MODE_EHSM_KMS = "EHSMKeyManagementService"
  val MODE_AZURE_KMS = "AzureKeyManagementService"
}

trait KeyManagementService extends Supportive {
  def retrievePrimaryKey(primaryKeySavePath: String, config: Configuration = null)
  def retrieveDataKey(primaryKeyPath: String, dataKeySavePath: String, config: Configuration = null)
  def retrieveDataKeyPlainText(primaryKeyPath: String, dataKeyPath: String,
                               config: Configuration = null): String
}
