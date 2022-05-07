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
    val result: JSONObject = new JSONObject(jsonObj.getString("result"))
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
