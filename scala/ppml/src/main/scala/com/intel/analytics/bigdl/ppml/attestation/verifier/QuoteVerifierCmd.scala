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

package com.intel.analytics.bigdl.ppml.attestation.verifier

import org.apache.logging.log4j.LogManager
import scopt.OptionParser

import java.io.{File, FileInputStream}
import java.math.BigInteger

import com.intel.analytics.bigdl.ppml.attestation._

object QuoteVerifierCmd {
    def main(args: Array[String]): Unit = {

        val logger = LogManager.getLogger(getClass)
        case class CmdParams(
            quoteOutputPath: String = "./quoteOutputDump"
        )

        val cmdParser =
            new OptionParser[CmdParams]("PPML Attestation Quote Verification Cmd tool") {
            opt[String]('q', "quote")
            .text("quoteOutputPath, default is ./quoteOutputDump")
            .action((x, c) => c.copy(quoteOutputPath = x))
        }

        val params = cmdParser.parse(args, CmdParams()).get

        val quoteOutputFile = new File(params.quoteOutputPath)
        if (quoteOutputFile.length == 0) {
            logger.error("Invalid quote file length.")
            throw new AttestationRuntimeException("Retrieving Gramine quote " +
              "returned Invalid file length!")
        }
        val in = new FileInputStream(quoteOutputFile)
        val quoteOutputData = new Array[Byte](quoteOutputFile.length.toInt)
        in.read(quoteOutputData)
        in.close()
        val result = verifyQuote(quoteOutputData)
    }

    def verifyQuote(quote: Array[Byte]): Int = {
        val number = new BigInteger(quote)
        val zero = new BigInteger("0")
        // If the quoteOutput is greater than 0,then 1 will be return,
        // if it is equal to 0, then 0 will be return,
        // else -1 will be return
        number.compareTo(zero)
    }

}
