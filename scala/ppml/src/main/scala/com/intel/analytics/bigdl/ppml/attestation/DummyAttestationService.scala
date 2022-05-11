package com.intel.analytics.bigdl.ppml.attestation

import java.math.BigInteger
import org.json.JSONObject
import scala.util.Random

class DummyAttestationService extends AttestationService {

    override def register(appID: String): String = "true"

    override def getPolicy(appID: String): String = "true"

    override def setPolicy(policy: JSONObject): String = "true"

    def getQuoteFromServer(): String = {
        "test"
    }

    override def attestWithServer(quote: String): (Boolean, String) = {
        timing("DummyAttestationService retrieveVerifyQuoteResult") {
            require(quote != null , "quote should be specified")
            val nonce: String="test"
            val response: JSONObject = new JSONObject()
            val number = new BigInteger(quote)
            val zero = new BigInteger("0")
            var verifyQuoteResult = true
            response.put("code", 200)
            response.put("message", "success")
            response.put("nonce", nonce)
            if(number.compareTo(zero) != 1) {
                verifyQuoteResult = false
            }
            response.put("result", verifyQuoteResult)
            val sign:String = (1 to 16).map(x => Random.nextInt(10)).mkString
            response.put("sign", sign)
            (verifyQuoteResult, response.toString)
        }
    }
}
