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


package com.intel.analytics.bigdl.ppml.attestation.service

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import org.apache.logging.log4j.LogManager
import org.json.JSONObject

import scala.util.Random

import com.intel.analytics.bigdl.ppml.attestation._


/**
 * Dummy Attestation Service for Test
 * If Quote String contains, "true" then return true
 */
class DummyAttestationService extends AttestationService {

    val logger = LogManager.getLogger(getClass)

    override def register(appID: String): String = "true"

    override def getPolicy(appID: String): String = "true"

    override def setPolicy(policy: JSONObject): String = "true"

    /**
     * Generate a quote randomly
     * @return a quote of String type
     */
    def getQuoteFromServer(challenge: String): String = {
        val userReportData = new Array[Byte](16)
        Random.nextBytes(userReportData)
        new String(userReportData)
    }

    /**
     * Do a quote verification
     * @param quote the quote generated before
     * @return the result and response of quote verify.
     *         If the quote contains the substring "true" then return true,
     *         else return false
     */
    override def attestWithServer(quote: String): (Boolean, String) = {
        timing("DummyAttestationService retrieveVerifyQuoteResult") {
            if (quote == null) {
                Log4Error.invalidInputError(false,
                    "Quote should be specified")
            }
            val nonce: String = "test"
            val response: JSONObject = new JSONObject()
            response.put("code", 200)
            response.put("message", "success")
            response.put("nonce", nonce)
            val verifyQuoteResult = quote.indexOf("true") >= 0
            response.put("result", verifyQuoteResult)
            val sign = (1 to 16).map(x => Random.nextInt(10)).mkString
            response.put("sign", sign)
            (verifyQuoteResult, response.toString)
        }
    }
    /**
     * Do a quote verification
     * @param quote the quote generated before
     * @param policyID a policy ID not used in dummy attestation service
     * @return the result and response of quote verify.
     *         If the quote contains the substring "true" then return true,
     *         else return false
     */
    override def attestWithServer(quote: String, policyID: String): (Boolean, String) = {
        attestWithServer(quote)
    }
}
