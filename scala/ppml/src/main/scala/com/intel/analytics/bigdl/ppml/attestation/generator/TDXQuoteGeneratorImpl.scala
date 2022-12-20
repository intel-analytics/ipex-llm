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

package com.intel.analytics.bigdl.ppml.attestation.generator

import com.intel.analytics.bigdl.ppml.dcap.Attestation

import org.apache.logging.log4j.LogManager

import java.io._

import com.intel.analytics.bigdl.ppml.attestation._

class TDXQuoteGeneratorImpl extends QuoteGenerator {

  val logger = LogManager.getLogger(getClass)

  @throws(classOf[AttestationRuntimeException])
  override def getQuote(userReportData: Array[Byte]): Array[Byte] = {

    if (userReportData.length == 0 || userReportData.length > 64) {
      logger.error("Incorrect userReport size!")
      throw new AttestationRuntimeException("Incorrect userReportData size!")
    }

    var quote = Array[Byte]()
    try {
      quote = Attestation.tdxGenerateQuote(userReportData)
    } catch {
      case e: Exception =>
        logger.error("Failed to get quote.")
        throw new AttestationRuntimeException("Failed to obtain quote " +
          "!", e)
    }

    return quote

    throw new AttestationRuntimeException("Unexpected workflow when generating TDX Quote!")
  }
}
