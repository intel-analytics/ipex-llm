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


package com.intel.analytics.bigdl.ppml.attestation

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.utils.EHSMParams
import com.intel.analytics.bigdl.ppml.utils.HTTPUtil.postRequest
import org.apache.logging.log4j.LogManager
import org.json.JSONObject

/**
 * Attestation Service provided by ehsm
 * @param kmsServerIP ehsm IP
 * @param kmsServerPort ehsm port
 * @param ehsmAPPID application ID
 * @param ehsmAPPKEY application Key
 */
class EHSMAttestationService(kmsServerIP: String, kmsServerPort: String,
                             ehsmAPPID: String, ehsmAPPKEY: String)
  extends AttestationService {

  val logger = LogManager.getLogger(getClass)

  // Quote
  val PAYLOAD_QUOTE = "quote"
  val PAYLOAD_NONCE = "nonce"
  val PAYLOAD_POLICYID = "policyId"

  val ACTION_VERIFY_QUOTE = "VerifyQuote"
  // Respone keys
  val RES_RESULT = "result"
  val RES_SIGN = "sign"

  override def register(appID: String): String = "true"

  override def getPolicy(appID: String): String = "true"

  override def setPolicy(policy: JSONObject): String = "true"

  def getQuoteFromServer(): String = {
    // TODO Get qutoe from ehsm
    "test"
  }

  override def attestWithServer(quote: String, policyId: String): (Boolean, String) = {
    // TODO nonce
    val nonce: String = "test"
    if (quote == null) {
      Log4Error.invalidInputError(false,
        "Quote should be specified")
    }
    if (policyId == null) {
      Log4Error.invalidInputError(false,
        "PolicyId should be specified")
    }
    val action: String = ACTION_VERIFY_QUOTE
    val currentTime = System.currentTimeMillis() // ms
    val timestamp = s"$currentTime"
    val ehsmParams = new EHSMParams(ehsmAPPID, ehsmAPPKEY, timestamp)
    ehsmParams.addPayloadElement(PAYLOAD_QUOTE, quote)
    ehsmParams.addPayloadElement(PAYLOAD_NONCE, nonce)
    ehsmParams.addPayloadElement(PAYLOAD_POLICYID, policyId)

    val postResult: JSONObject = timing("EHSMKeyManagementService request for VerifyQuote") {
      val postString: String = ehsmParams.getPostJSONString()
      postRequest(constructUrl(action), postString)
    }
    // Check sign with nonce
    val sign = postResult.getString(RES_SIGN)
    val verifyQuoteResult = postResult.getBoolean(RES_RESULT)
    (verifyQuoteResult, postResult.toString)
  }

  private def constructUrl(action: String): String = {
    s"http://$kmsServerIP:$kmsServerPort/ehsm?Action=$action"
  }
}
