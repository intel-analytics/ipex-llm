package com.intel.analytics.bigdl.ppml.kms

import scala.collection.mutable.HashMap
import scala.util.Random
import com.intel.analytics.bigdl.ppml.utils.KeyReaderWriter

class SimpleKeyManagementService(simpleAPPID:String = "", simpleAPPKEY:String = "") extends KeyManagementService {
  var enrollMap:HashMap[String,String] = new HashMap[String,String]
  val keyReaderWriter = new KeyReaderWriter
  
  if (simpleAPPID != "" && simpleAPPID != "") {
    setAppIdAndKey(simpleAPPID, simpleAPPKEY)
  } else {
    enroll()
  }

  def enroll():(String, String) = {
    val appid:String = (1 to 12).map(x => Random.nextInt(10)).mkString
    val appkey:String = (1 to 12).map(x => Random.nextInt(10)).mkString
    enrollMap.contains(appid) match {
      case true => setAppIdAndKey(appid, appkey) //TODO
      case false => setAppIdAndKey(appid, appkey)
    }
    (appid, appkey)
  }

  def retrievePrimaryKey(primaryKeySavePath: String) = {
    timing("SimpleKeyManagementService retrievePrimaryKey") {
      require(enrollMap.keySet.contains(_appid) && enrollMap(_appid) == _appkey, "appid and appkey do not match!")
      require(primaryKeySavePath != null && primaryKeySavePath != "", "primaryKeySavePath should be specified")
      val suffix:String = (1 to 4).map { x => Random.nextInt(10) }.mkString
      val encryptedPrimaryKey:String = _appid + suffix
      keyReaderWriter.writeKeyToFile(primaryKeySavePath, encryptedPrimaryKey)
    }
  }

  def retrieveDataKey(primaryKeyPath: String, dataKeySavePath: String) = {
    timing("SimpleKeyManagementService retrieveDataKey") {
      require(enrollMap.keySet.contains(_appid) && enrollMap(_appid) == _appkey, "appid and appkey do not match!")
      require(primaryKeyPath != null && primaryKeyPath != "", "primaryKeyPath should be specified")
      require(dataKeySavePath != null && dataKeySavePath != "", "dataKeySavePath should be specified")
      val primaryKeyPlaintext:String = keyReaderWriter.readKeyFromFile(primaryKeyPath)
      require(primaryKeyPlaintext.substring(0, 12) == _appid, "appid and primarykey should be matched!")
      val randVect = (1 to 16).map { x => Random.nextInt(10) }
      val dataKeyPlaintext:String = randVect.mkString
      var dataKeyCiphertext:String = ""
      for(i <- 0 until 16){
        dataKeyCiphertext += '0' + ((primaryKeyPlaintext(i) - '0') + (dataKeyPlaintext(i) - '0')) % 10
      }
      keyReaderWriter.writeKeyToFile(dataKeySavePath, dataKeyCiphertext)
    }
  }

  def retrieveDataKeyPlainText(primaryKeyPath: String, dataKeyPath: String): String = {
    timing("SimpleKeyManagementService retrieveDataKeyPlaintext") {
      require(enrollMap.keySet.contains(_appid) && enrollMap(_appid) == _appkey, "appid and appkey do not match!")
      require(primaryKeyPath != null && primaryKeyPath != "", "primaryKeyPath should be specified")
      require(dataKeyPath != null && dataKeyPath != "", "dataKeyPath should be specified")
      val primaryKeyCiphertext:String = keyReaderWriter.readKeyFromFile(primaryKeyPath)
      require(primaryKeyCiphertext.substring(12) == _appid, "appid and primarykey should be matched!")
      val dataKeyCiphertext:String = keyReaderWriter.readKeyFromFile(dataKeyPath)
      var dataKeyPlaintext:String = ""
      for(i <- 0 until 16){
        dataKeyPlaintext += '0' + ((dataKeyCiphertext(i) - '0') - (primaryKeyCiphertext(i) - '0') + 10) % 10
      }
      dataKeyPlaintext
    }
  }

  private def setAppIdAndKey(appid:String, appkey:String) = {
    _appid = appid
    _appkey = appkey
    enrollMap(_appid) = _appkey
  }

}

object SimpleKeyManagementService {
  def apply(): SimpleKeyManagementService = {
    new SimpleKeyManagementService()
  }
}
