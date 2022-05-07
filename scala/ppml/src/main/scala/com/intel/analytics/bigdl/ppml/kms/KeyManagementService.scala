package com.intel.analytics.bigdl.ppml.kms

import com.intel.analytics.bigdl.ppml.utils.Supportive

object KMS_CONVENTION {
  val MODE_SIMPLE_KMS = "SimpleKeyManagementService"
  val MODE_EHSM_KMS = "EHSMKeyManagementService"
}

trait KeyManagementService extends Supportive {
  var _appid:String = _
  var _appkey:String = _

  def enroll():(String,String)
  def retrievePrimaryKey(primaryKeySavePath: String)
  def retrieveDataKey(primaryKeyPath: String, dataKeySavePath: String)
  def retrieveDataKeyPlainText(primaryKeyPath: String, dataKeyPath: String): String
}
