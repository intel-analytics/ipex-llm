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


package com.intel.analytics.bigdl.ppml.kms

import org.apache.hadoop.conf.Configuration
import com.intel.analytics.bigdl.dllib.utils.Log4Error

import javax.net.ssl.SSLContext
import org.apache.http.conn.ssl.AllowAllHostnameVerifier
import org.apache.http.conn.ssl.SSLConnectionSocketFactory
import org.apache.http.ssl.SSLContextBuilder
import org.apache.http.ssl.SSLContexts
import javax.net.ssl.X509TrustManager
import java.security.cert.X509Certificate
import javax.net.ssl.TrustManager
import org.apache.http.util.EntityUtils
import java.security.SecureRandom
import org.slf4j.LoggerFactory

import org.apache.http.impl.client.CloseableHttpClient
import org.apache.http.impl.client.HttpClients
import org.apache.http.client.methods.HttpPost
import org.apache.http.client.methods.HttpGet
import org.apache.http.impl.client.HttpClients

object BIGDLKMS_ACTION extends Enumeration {
  type TYPE = Value
  val CREATE_USER, CREATE_PRIMARY_KEY, CREATE_DATA_KEY, GET_DATA_KEY = Value
  val POST_REQUEST, GET_REQUEST = Value
}

class BigDLKeyManagementService(
      kmsServerIP: String,
      kmsServerPort: String,
      userName: String,
      userToken: String)extends KeyManagementService {

  Log4Error.invalidInputError(userName != null && userName != "",
    """
    User name should not be empty string.
    Pre-create or use name and token to enroll a new one.
    """
    )
  Log4Error.invalidInputError(userToken != null && userToken != "",
    """
    User token should not be empty string.
    Pre-create or use name and token to enroll a new one.
    """
    )
  val logger = LoggerFactory.getLogger(getClass)
  val sslConSocFactory = {
    val sslContext: SSLContext = SSLContext.getInstance("SSL")
    val trustManager: TrustManager = new X509TrustManager() {
      override def checkClientTrusted(chain: Array[X509Certificate], authType: String): Unit = {}
      override def checkServerTrusted(chain: Array[X509Certificate], authType: String): Unit = {}
      override def getAcceptedIssuers(): Array[X509Certificate] = Array.empty
    }
    sslContext.init(null, Array(trustManager), new SecureRandom())
    new SSLConnectionSocketFactory(sslContext, new AllowAllHostnameVerifier())
  }

  def enroll(): Unit = {
    // call enroll if user and token has not been created before
    val action = BIGDLKMS_ACTION.CREATE_USER
    val url = constructBaseUrl(action, userName)
    val response = timing("BKeyManagementService request for creating user") {
      sendRequest(BIGDLKMS_ACTION.POST_REQUEST, url)
    }
    logger.info(response)
  }

  def retrievePrimaryKey(primaryKeyName: String, config: Configuration = null): Unit = {
    Log4Error.invalidInputError(primaryKeyName != null && primaryKeyName != "",
      "primaryKeyName should be specified")
    logger.info("BigDLKMS retrievePrimaryKey API create a primary key at KMS server" +
                " and do not save locally.")
    val action = BIGDLKMS_ACTION.CREATE_PRIMARY_KEY
    val url = constructBaseUrl(action, primaryKeyName) + s"&user=$userName"
    val response = timing("BKeyManagementService request for creating primaryKey") {
      sendRequest(BIGDLKMS_ACTION.POST_REQUEST, url)
    }
    logger.info(response)
  }

  def retrieveDataKey(primaryKeyName: String, dataKeyName: String,
                      config: Configuration = null): Unit = {
    Log4Error.invalidInputError(primaryKeyName != null && primaryKeyName != "",
      "primaryKeyName should be specified")
    Log4Error.invalidInputError(dataKeyName != null && dataKeyName != "",
      "dataKeyName should be specified")
    logger.info("BigDLKMS retrieveDataKey API create a data key at KMS server" +
                " and do not save locally.")
    val action = BIGDLKMS_ACTION.CREATE_DATA_KEY
    val url = constructBaseUrl(action, dataKeyName) +
              s"&user=$userName&primaryKeyName=$primaryKeyName"
    val response = timing("BKeyManagementService request for creating dataKey") {
      sendRequest(BIGDLKMS_ACTION.POST_REQUEST, url)
    }
    logger.info(response)
  }


  def retrieveDataKeyPlainText(primaryKeyName: String, dataKeyName: String,
                                        config: Configuration = null): String = {
    Log4Error.invalidInputError(primaryKeyName != null && primaryKeyName != "",
      "primaryKeyName should be specified")
    Log4Error.invalidInputError(dataKeyName != null && dataKeyName != "",
      "dataKeyName should be specified")
    logger.info("BigDLKMS retrieveDataKeyPlaintext API get the specific data key from KMS server")
    val action = BIGDLKMS_ACTION.GET_DATA_KEY
    val url = constructBaseUrl(action, dataKeyName) +
              s"&user=$userName&primaryKeyName=$primaryKeyName"
    val response = timing("BigDLKeyManagementService request for getting dataKey") {
      sendRequest(BIGDLKMS_ACTION.GET_REQUEST, url)
    }
    response
  }


  private def constructBaseUrl(action: BIGDLKMS_ACTION.TYPE, customParamName: String): String = {
    val path = action match {
        case BIGDLKMS_ACTION.CREATE_USER => "/user/"
        case BIGDLKMS_ACTION.CREATE_PRIMARY_KEY => "/primaryKey/"
        case BIGDLKMS_ACTION.CREATE_DATA_KEY => "/dataKey/"
        case BIGDLKMS_ACTION.GET_DATA_KEY => "/dataKey/"
    }
    val baseUrl = s"https://$kmsServerIP:$kmsServerPort/" +
                  path + customParamName + s"?token=$userToken"
    baseUrl
  }

  private def sendRequest(requestType: BIGDLKMS_ACTION.TYPE, url: String): String = {
    val clientbuilder = HttpClients.custom().setSSLSocketFactory(sslConSocFactory)
    val httpsClient: CloseableHttpClient = clientbuilder.build()
    val request = requestType match {
        case BIGDLKMS_ACTION.POST_REQUEST => new HttpPost(url)
        case BIGDLKMS_ACTION.GET_REQUEST => new HttpGet(url)
    }
    val response = httpsClient.execute(request)
    EntityUtils.toString(response.getEntity, "UTF-8")
  }

}
