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

package com.intel.analytics.bigdl.ppml.utils

import java.io.Serializable
import scala.collection.mutable.HashMap
import com.intel.analytics.bigdl.ppml.kms.KeyManagementService
import com.intel.analytics.bigdl.dllib.utils.Log4Error

class KMSManagement extends Serializable {
    var multiKms = new HashMap[String, KeyManagementService]
    def enrollKms(name: String, kms: KeyManagementService): Unit = {
        Log4Error.invalidInputError(!(multiKms.contains(name)),
                                    s"KMSs with name $name are replicated.")
        multiKms += (name -> kms)
    }
    def getKms(name: String): KeyManagementService = {
        Log4Error.invalidInputError(multiKms.contains(name),
                                    s"cannot get a not-existing kms.")
        multiKms.get(name).get
    }
}

