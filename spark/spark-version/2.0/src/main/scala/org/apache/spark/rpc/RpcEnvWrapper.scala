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

package org.apache.spark.rpc

import org.apache.spark.util.RpcUtils
import org.apache.spark.{SecurityManager, SparkConf}

object RpcEnvWrapper {
  def create(
    name: String,
    host: String,
    port: Int,
    conf: SparkConf,
    securityManager: SecurityManager,
    clientMode: Boolean = false): RpcEnv = {
    println("hehe")
    RpcEnv.create(name, host, port, conf, securityManager)
  }

  def setupEndpoint(rpcEnv: RpcEnv, name: String, endpoint: RpcEndpoint): RpcEndpointRef = {
    rpcEnv.setupEndpoint(name, endpoint)
  }

  def setupEndpointRef(rpcEnv: RpcEnv, systemName: String, rpcAddress: RpcAddress,
    endpointName: String): RpcEndpointRef = {
    rpcEnv.setupEndpointRef(rpcAddress, endpointName)
  }
}

