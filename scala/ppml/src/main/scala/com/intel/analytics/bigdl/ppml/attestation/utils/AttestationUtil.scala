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
package com.intel.analytics.bigdl.ppml.attestation.utils

import java.math.BigInteger
import java.security.SecureRandom
import org.json4s.jackson.Serialization
import org.json4s._

import scala.util.parsing.json._
import com.intel.analytics.bigdl.ppml.attestation.Policy

object AttestationUtil {
  implicit val formats = DefaultFormats

  def mapToString(map: Map[String, Any]): String = {
    Serialization.write(map)
  }

  def strMapToString(map: Map[String, String]): String = {
    Serialization.write(map)
  }

  def policyMapToString(map: Map[String, Policy]): String = {
    Serialization.write(map)
  }

  def stringToMap(str: String): Map[String, Any] = {
    JSON.parseFull(str) match {
      case Some(map: Map[String, Any]) => map
      case None => Map.empty
    }
  }

  def stringToStrMap(str: String): Map[String, String] = {
    JSON.parseFull(str) match {
      case Some(map: Map[String, String]) => map
      case None => Map.empty
    }
  }

  def stringToPolicyMap(str: String): Map[String, Policy] = {
    JSON.parseFull(str) match {
      case Some(map: Map[String, Policy]) => map
      case None => Map.empty
    }
  }

  def getMREnclaveFromQuote(quote: Array[Byte]): String = {
    new BigInteger(1, quote.slice(112, 144)).toString(16)
  }

  def getMRSignerFromQuote(quote: Array[Byte]): String = {
    new BigInteger(1, quote.slice(176, 208)).toString(16)
  }

  def getISVProdIDFromQuote(quote: Array[Byte]): String = {
    new BigInteger(1, quote.slice(304, 306)).toString(16)
  }

  def generateToken(length: Int = 32): String = {
    val secureRandom = new SecureRandom()
    val charSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    val token = new Array[Char](length)
    for (i <- 0 until length) {
      token(i) = charSet.charAt(secureRandom.nextInt(charSet.length))
    }
    token.mkString
  }
}
