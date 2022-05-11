package com.intel.analytics.bigdl.ppml.attestation
import com.intel.analytics.bigdl.ppml.utils.EHSMParams
import com.intel.analytics.bigdl.ppml.utils.HTTPUtil.postRequest
import org.json.JSONObject

class EHSMAttestationService(kmsServerIP: String, kmsServerPort: String, ehsmAPPID: String, ehsmAPPKEY: String)
  extends AttestationService {

  // Quote
  val PAYLOAD_QUOTE = "quote"
  val PAYLOAD_NONCE = "nonce"

  val ACTION_VERIFY_QUOTE = "VerifyQuote"
  // Respone keys
  val RES_RESULT = "result"
  val RES_SIGN = "sign"

  override def register(appID: String): String = "true"

  override def getPolicy(appID: String): String = "true"

  override def setPolicy(policy: JSONObject): String = "true"

  def getQuoteFromServer(): String = {
    "test"
  }

  override def attestWithServer(quote: String): (Boolean, String) = {
    // TODO nonce
    val nonce: String="test"
    require(quote != null , "quote should be specified")
    val action: String = ACTION_VERIFY_QUOTE
    val currentTime = System.currentTimeMillis() // ms
    val timestamp = s"$currentTime"
    val ehsmParams = new EHSMParams(ehsmAPPID, ehsmAPPKEY, timestamp)
    ehsmParams.addPayloadElement(PAYLOAD_QUOTE, quote)
    ehsmParams.addPayloadElement(PAYLOAD_NONCE, nonce)
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
