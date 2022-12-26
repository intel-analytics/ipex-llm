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


package com.intel.analytics.bigdl.ppml.attestation

import org.apache.logging.log4j.LogManager
import scopt.OptionParser

import java.util.Base64

import com.intel.analytics.bigdl.ppml.attestation.service._
import com.intel.analytics.bigdl.ppml.attestation.generator._
import com.intel.analytics.bigdl.ppml.attestation.verifier._

object TdxQuoteGenerate {
    def main(args: Array[String]): Unit = {

        val logger = LogManager.getLogger(getClass)
        case class CmdParams(userReport: String = "test")
        val cmdParser: OptionParser[CmdParams] = new
            OptionParser[CmdParams]("PPML Quote Generation Cmd tool") {
            opt[String]('r', "userReport")
              .text("userReport")
              .action((x, c) => c.copy(userReport = x))
        }
        val params = cmdParser.parse(args, CmdParams()).get
        val userReportData = params.userReport

        val quoteGenerator = new TDXQuoteGeneratorImpl()
        val quote = quoteGenerator.getQuote(userReportData.getBytes)
        val res = new String(quote)
        System.out.println(res)

        System.exit(0)
    }

}
