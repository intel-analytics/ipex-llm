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

import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, EncryptRuntimeException, PLAIN_TEXT}
import com.intel.analytics.bigdl.ppml.kms.{EHSMKeyManagementService, KMS_CONVENTION, SimpleKeyManagementService}

import java.io.File

case class EncryptIOArguments(
                               inputPath: String = "./input",
                               outputPath: String = "./output",
                               inputEncryptMode: CryptoMode = PLAIN_TEXT,
                               outputEncryptMode: CryptoMode = PLAIN_TEXT,
                               inputPartitionNum: Int = 4,
                               outputPartitionNum: Int = 4,
                               primaryKeyPath: String = "./primaryKeyPath",
                               dataKeyPath: String = "./dataKeyPath",
                               kmsType: String = KMS_CONVENTION.MODE_SIMPLE_KMS,
                               kmsServerIP: String = "0.0.0.0",
                               kmsServerPort: String = "5984",
                               ehsmAPPID: String = "ehsmAPPID",
                               ehsmAPIKEY: String = "ehsmAPIKEY",
                               simpleAPPID: String = "simpleAPPID",
                               simpleAPIKEY: String = "simpleAPIKEY",
                               keyVaultName: String = "keyVaultName",
                               managedIdentityClientId: String = "",
                               userName: String = "bigdlKMSUserName",
                               userToken: String = "bigdlKMSUserToken") {
  def ppmlArgs(): Map[String, String] = {
    val kmsArgs = scala.collection.mutable.Map[String, String]()
    kmsArgs("spark.bigdl.kms.type") = kmsType
    kmsType match {
      case KMS_CONVENTION.MODE_EHSM_KMS =>
        kmsArgs("spark.bigdl.kms.ip") = kmsServerIP
        kmsArgs("spark.bigdl.kms.port") = kmsServerPort
        kmsArgs("spark.bigdl.kms.appId") = ehsmAPPID
        kmsArgs("spark.bigdl.kms.apiKey") = ehsmAPIKEY
      case KMS_CONVENTION.MODE_SIMPLE_KMS =>
        kmsArgs("spark.bigdl.kms.appId") = simpleAPPID
        kmsArgs("spark.bigdl.kms.apiKey") = simpleAPIKEY
      case KMS_CONVENTION.MODE_AZURE_KMS =>
        kmsArgs("spark.bigdl.kms.vault") = keyVaultName
        kmsArgs("spark.bigdl.kms.clientId") = managedIdentityClientId
      case KMS_CONVENTION.MODE_BIGDL_KMS =>
        kmsArgs("spark.bigdl.kms.ip") = kmsServerIP
        kmsArgs("spark.bigdl.kms.port") = kmsServerPort
        kmsArgs("spark.bigdl.kms.user") = userName
        kmsArgs("spark.bigdl.kms.token") = userToken
      case _ =>
        throw new EncryptRuntimeException("Wrong kms type")
    }
    if (new File(primaryKeyPath).exists()) {
      kmsArgs("spark.bigdl.kms.primaryKey") = primaryKeyPath
    }
    if (new File(dataKeyPath).exists()) {
      kmsArgs("spark.bigdl.kms.dataKey") = dataKeyPath
    }
    kmsArgs.toMap
  }
}

object EncryptIOArguments {
  val parser = new scopt.OptionParser[EncryptIOArguments]("BigDL PPML E2E workflow") {
    head("BigDL PPML E2E workflow")
    opt[String]('i', "inputPath")
      .action((x, c) => c.copy(inputPath = x))
      .text("inputPath")
    opt[String]('o', "outputPath")
      .action((x, c) => c.copy(outputPath = x))
      .text("outputPath")
    opt[String]('a', "inputEncryptModeValue")
      .action((x, c) => c.copy(inputEncryptMode = CryptoMode.parse(x)))
      .text("inputEncryptModeValue: plain_text/aes_cbc_pkcs5padding")
    opt[String]('b', "outputEncryptModeValue")
      .action((x, c) => c.copy(outputEncryptMode = CryptoMode.parse(x)))
      .text("outputEncryptModeValue: plain_text/aes_cbc_pkcs5padding")
    opt[String]('c', "outputPath")
      .action((x, c) => c.copy(outputPath = x))
      .text("outputPath")
    opt[Int]('f', "inputPartitionNum")
      .action((x, c) => c.copy(inputPartitionNum = x))
      .text("inputPartitionNum")
    opt[Int]('e', "outputPartitionNum")
      .action((x, c) => c.copy(outputPartitionNum = x))
      .text("outputPartitionNum")
    opt[String]('p', "primaryKeyPath")
      .action((x, c) => c.copy(primaryKeyPath = x))
      .text("primaryKeyPath")
    opt[String]('d', "dataKeyPath")
      .action((x, c) => c.copy(dataKeyPath = x))
      .text("dataKeyPath")
    opt[String]('k', "kmsType")
      .action((x, c) => c.copy(kmsType = x))
      .text("kmsType")
    opt[String]('g', "kmsServerIP")
      .action((x, c) => c.copy(kmsServerIP = x))
      .text("kmsServerIP")
    opt[String]('h', "kmsServerPort")
      .action((x, c) => c.copy(kmsServerPort = x))
      .text("kmsServerPort")
    opt[String]('j', "ehsmAPPID")
      .action((x, c) => c.copy(ehsmAPPID = x))
      .text("ehsmAPPID")
    opt[String]('k', "ehsmAPIKEY")
      .action((x, c) => c.copy(ehsmAPIKEY = x))
      .text("ehsmAPIKEY")
    opt[String]('s', "simpleAPPID")
      .action((x, c) => c.copy(simpleAPPID = x))
      .text("simpleAPPID")
    opt[String]('k', "simpleAPIKEY")
      .action((x, c) => c.copy(simpleAPIKEY = x))
      .text("simpleAPIKEY")
    opt[String]('v', "vaultName")
      .action((x, c) => c.copy(keyVaultName = x))
      .text("keyVaultName")
    opt[String]('u', "clientId")
      .action((x, c) => c.copy(managedIdentityClientId = x))
      .text("keyVaultName")
    opt[String]('z', "userName")
      .action((x, c) => c.copy(userName = x))
      .text("bigdlKMSUserName")
    opt[String]('y', "userToken")
      .action((x, c) => c.copy(userToken = x))
      .text("bigdlKMSUserToken")
  }
}
