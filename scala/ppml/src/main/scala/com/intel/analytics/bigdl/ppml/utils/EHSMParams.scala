package com.intel.analytics.bigdl.ppml.utils

import scala.collection.mutable.HashMap
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import java.util.Base64

class EHSMParams(appid: String, appkey: String, timestamp: String) {

  var payload = new HashMap[String, String]

  def addPayloadElement(payloadElementKey: String, payloadElementVal: String) = {
    payload(payloadElementKey) = payloadElementVal
  }

  def getPostJSONString(): String = {

    var postJSONString: String = "{\"appid\":\"" + appid + "\""
    postJSONString = postJSONString + ",\"payload\":{"
    for ((payloadElementKey, payloadElementVal) <- payload){
      if(payloadElementKey == "keylen"){
        postJSONString = postJSONString + "\"" + payloadElementKey + "\":" + payloadElementVal + ','
      }else{
        postJSONString = postJSONString + "\"" + payloadElementKey + "\":\"" + payloadElementVal + "\","
      }
    }
    postJSONString = postJSONString.dropRight(1)
    val signCiphertextString: String = getSignCiphertextString()
    postJSONString = postJSONString + "},\"timestamp\":\"" + timestamp + "\",\"sign\":\"" + signCiphertextString + "\"}"
    postJSONString
  }


  private def getSignCiphertextString(): String = {
    val secret = new SecretKeySpec(appkey.getBytes("UTF-8"), "SHA256")
    val mac = Mac.getInstance("HmacSHA256")
    mac.init(secret)
    val signPlaintextString: String = getSignPlaintextString()
    val signCiphertextString: String = Base64.getEncoder.encodeToString(mac.doFinal(signPlaintextString.getBytes("UTF-8")))
    signCiphertextString
  }

  private def getSignPlaintextString(): String = {

    require(appid != "" && appkey != "" && timestamp != "" && !payload.isEmpty, "Lack necessary param or payload!")
    var signString: String = s"appid=$appid&payload="
    val tmp = Map(payload.toSeq.sortWith(_._1 < _._1):_*)
    for ((payloadElementKey, payloadElementVal) <- tmp) signString = signString + s"$payloadElementKey=$payloadElementVal&"
    signString = signString + s"timestamp=$timestamp"
    signString
  }

}

