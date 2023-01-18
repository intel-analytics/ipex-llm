package com.intel.analytics.bigdl.ppml.utils

import java.io.Serializable
import com.intel.analytics.bigdl.ppml.crypto.{CryptoMode, AES_CBC_PKCS5PADDING}
import scala.collection.mutable.HashMap
import com.intel.analytics.bigdl.dllib.utils.Log4Error

trait DataStore extends Serializable {
    def kmsName: String
    def path: Path
    def primaryKey: String
    def dataKey: String
    def encryptMode: CryptoMode
}

case class DataSource(
    val kmsName: String,
    val path: String,
    val primaryKey: String,
    val dataKey: String,
    val encryptMode: CryptoMode = AES_CBC_PKCS5PADDING
) extends DataStore

case class DataSink(
    val kmsName: String,
    val path: String,
    val primaryKey: String,
    val dataKey: String,
    val encryptMode: CryptoMode = AES_CBC_PKCS5PADDING
) extends DataStore 

class DataStoreManagement extends Serializable {
    var dataStores = new HashMap[String, DataStore]

    def enrollDataStore(name: String, dataStore: DataStore): Unit = {
        Log4Error.invalidInputError(!(dataStores.contains(name)),
                                    s"dataStore with name $name are replicated.")
        dataStores += (name -> dataStore)
    }

    def getDataStore(name: String): DataStore = {
        Log4Error.invalidInputError(dataStores.contains(name),
                                    s"cannot get a not-existing dataSource.")
        dataStores.get(name).get
    }

}
