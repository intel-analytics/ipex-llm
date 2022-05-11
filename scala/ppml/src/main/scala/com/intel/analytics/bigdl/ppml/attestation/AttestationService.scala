package com.intel.analytics.bigdl.ppml.attestation

import com.intel.analytics.bigdl.ppml.utils.Supportive
import org.json.JSONObject


object ATTESTATION_CONVENTION {
  val MODE_DUMMY = "DummyAttestationService"
  val MODE_EHSM_KMS = "EHSMAttestationService"
  val MODE_AMBER = "AMBERService"
  val MODE_AZURE = "AzureAttestationService"
}

trait AttestationService extends Supportive {
  // Admin
  // TODO split to admin API
  def register(appID: String): String
  def getPolicy(appID: String): String
  def setPolicy(policy: JSONObject): String
  // App
  def getQuoteFromServer(): String
  def attestWithServer(quote: String): (Boolean, String)
}
