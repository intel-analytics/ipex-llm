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

import com.intel.analytics.bigdl.ppml.utils.Supportive
import org.json.JSONObject


object ATTESTATION_CONVENTION {
  val MODE_DUMMY = "DummyAttestationService"
  val MODE_EHSM_KMS = "EHSMAttestationService"
  val MODE_AMBER = "AMBERService"
  val MODE_AZURE = "AzureAttestationService"
}

/**
 * Attestation interface.
 */
trait AttestationService extends Supportive {
  // Admin
  // TODO split to admin API

  /**
   * Register application.
   * @param appID the APPId of your application.
   * @return the result of registration.
   */
  def register(appID: String): String

  /**
   * Get policy.
   * @param appID the APPId of your application.
   * @return the policy.
   */
  def getPolicy(appID: String): String

  /**
   * Set policy.
   * @param policy the policy.
   * @return the result of setting policy.
   */
  def setPolicy(policy: JSONObject): String
  // App

  /**
   * Get Quote from Attestation Service
   * @return quote in string
   */
  def getQuoteFromServer(): String

  /**
   * Send quote to Attestation Service, get attestation result
   * @param quote application's quote
   * @return attestation result/token
   */
  def attestWithServer(quote: String): (Boolean, String)
}
