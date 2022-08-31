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

import org.apache.http.client.methods.HttpPost
import org.apache.http.entity.StringEntity
import org.apache.http.impl.client.HttpClients
import org.apache.http.message.BasicHeader
import org.apache.http.util.EntityUtils
import org.json.JSONObject

object HTTPUtil {
  def postRequest(url: String, postString: String): JSONObject = {
    val response: String = retrieveResponse(url, postString)
    val jsonObj: JSONObject = new JSONObject(response)
    val result: JSONObject = jsonObj.getJSONObject("result")
    result
  }

  def retrieveResponse(url: String, params: String = null): String = {
    val httpClient = HttpClients.createDefault()
    val post = new HttpPost(url)
    post.setHeader(new BasicHeader("Content-Type", "application/json"));
    if (params != null) {
      post.setEntity(new StringEntity(params, "UTF-8"))
    }
    val response = httpClient.execute(post)
    EntityUtils.toString(response.getEntity, "UTF-8")
  }

}
