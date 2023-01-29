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
package com.intel.analytics.bigdl.ppml.fl.example.ckks

import com.intel.analytics.bigdl.ckks.CKKS

import java.io.File

// TODO: key should be provided by Key Management System
object GenerateCkksSecret {
  def main(args: Array[String]): Unit = {
    if (args.length >= 1) {
      val ckks = new CKKS()
      val keys = ckks.createSecrets()
      val computeKeys = new Array[Array[Byte]](2)
      computeKeys(0) = keys(0)
      computeKeys(1) = keys(2)
      val fullSecretFileName = args(0) + File.separator + "all_secret"
      val computeSecretFileName = args(0) + File.separator + "compute_secret"
      CKKS.saveSecret(keys, fullSecretFileName)
      println("save all secret to " + fullSecretFileName)
      CKKS.saveSecret(computeKeys, computeSecretFileName)
      println("save compute secret to " + computeSecretFileName)
    } else {
      println("please provide a path to save secret.")
    }
  }
}
