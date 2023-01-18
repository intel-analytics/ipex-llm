package com.intel.analytics.bigdl.ppml.kms

import java.io.Serializable
import scala.collection.mutable.HashMap
import com.intel.analytics.bigdl.ppml.kms.KeyManagementService
import com.intel.analytics.bigdl.dllib.utils.Log4Error

class KMSManagement extends Serializable {
    var multiKMS = new HashMap[String, KeyManagementService]
    
    def enrollKMS(name: String, kms: KeyManagementService): Unit = {
        Log4Error.invalidInputError(!(multiKms.contains(name)),
                                    s"KMSs with name $name are replicated.")
        multiKms += (name -> kms)
    }
    
    def getKMS(name: String): KeyManagementService = {
        Log4Error.invalidInputError(multiKms.contains(name),
                                    s"cannot get a not-existing kms.")
        multiKms.get(name).get
    }
}
