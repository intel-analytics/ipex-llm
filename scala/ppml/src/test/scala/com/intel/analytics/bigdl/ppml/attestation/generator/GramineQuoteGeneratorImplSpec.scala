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

import org.scalatest.{FlatSpec, Matchers}
import java.io.{FileOutputStream, FileInputStream, BufferedInputStream, BufferedOutputStream}
import scala.io.Source
import scala.util.Random
import sys.env

class GramineQuoteGeneratorImplSpec extends FlatSpec with Matchers {
  // get the 'envFlag' from the shell environments, enable TEE by 'export envFlag=TEE'
  val envFlag = if (env.contains("envFlag")) {
    env("envFlag").toString
  } else {
    "nonTEE"
  }

  // GramineQuoteGeneratorImplSpec
  "Gramine get Quote " should "work" in {
    if (envFlag=="TEE") {
      val gramineQuoteGenerator = new GramineQuoteGeneratorImpl()
      // generate a random userReportData.
      val userReportData = new Array[Byte](32)
      Random.nextBytes(userReportData)
      val quote = gramineQuoteGenerator.getQuote(userReportData)
      val quoteWriter = new FileOutputStream("gramine-quote-dump")
      quoteWriter.write(quote)
      quoteWriter.close()

    }
    else {
      val quote = Array[Byte](0x21, 0x21, 0x21, 0x21, 0x21, 0x21, 0x21)
      val quoteWriter = new FileOutputStream("dummy-gramine-quote-dump")
      quoteWriter.write(quote)
      quoteWriter.close()

    }
  }
}
