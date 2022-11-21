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

import org.apache.logging.log4j.LogManager

import java.io._

import com.intel.analytics.bigdl.ppml.attestation._
/**
 * QuoteGenerator for ppml-Occlum
 * (https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/scala/docker-occlum)
 */
class OcclumQuoteGeneratorImpl extends QuoteGenerator {

  val logger = LogManager.getLogger(getClass)
  val QUOTE_PATH = "/etc/occlum_attestation/quote"

  @throws(classOf[AttestationRuntimeException])
  override def getQuote(userReportData: Array[Byte]): Array[Byte] = {

    if (userReportData.length == 0 || userReportData.length > 32) {
      logger.error("Incorrect userReport size!")
      throw new AttestationRuntimeException("Incorrect userReportData size!")
    }


    // userReport will be write first by running cmd: occlum run /bin/dcap_c_test

    try {
      // read quote
      val quoteFile = new File(QUOTE_PATH)
      val in = new FileInputStream(quoteFile)
      val bufIn = new BufferedInputStream(in)
      val quote = Iterator.continually(bufIn.read()).takeWhile(_ != -1).map(_.toByte).toArray
      bufIn.close()
      in.close()
      if (quote.length == 0) {
        logger.error("Invalid quote file length.")
        throw new AttestationRuntimeException("Retrieving Occlum quote " +
          "returned Invalid file length!")
      }
      return quote
    } catch {
      case e: Exception =>
        logger.error("Failed to get quote.")
        throw new AttestationRuntimeException("Failed to obtain quote " +
          "content from file to buffer!", e)
    }

    throw new AttestationRuntimeException("Unexpected workflow when generating Occlum Quote!")
  }
}
