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

import java.io.{BufferedOutputStream, BufferedInputStream};
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import scala.collection.Iterator

import java.util.Base64

import com.intel.analytics.bigdl.ppml.attestation.service._
import com.intel.analytics.bigdl.ppml.attestation.verifier._

object TdxQuoteVerification {
    def main(args: Array[String]): Unit = {

        val logger = LogManager.getLogger(getClass)
        case class CmdParams(quote: String = "test")
        val cmdParser = new OptionParser[CmdParams]("PPML Quote Verification Cmd tool") {
            opt[String]('i', "quote")
              .text("quote")
              .action((x, c) => c.copy(quote = x))
        }
        val params = cmdParser.parse(args, CmdParams()).get
        val quotePath = params.quote
        var quote = Array[Byte]()
        try {
            // read quote
            val quoteFile = new File(quotePath)
            val in = new FileInputStream(quoteFile)
            val bufIn = new BufferedInputStream(in)
            quote = Iterator.continually(bufIn.read()).takeWhile(_ != -1).map(_.toByte).toArray
            bufIn.close()
            in.close()
            if (quote.length == 0) {
                logger.error("Invalid quote file length.")
                throw new AttestationRuntimeException("Retrieving Gramine quote " +
                "returned Invalid file length!")
            }
        } catch {
            case e: Exception =>
                logger.error("Failed to get quote.")
                throw new AttestationRuntimeException("Failed to obtain quote " +
                "content from file to buffer!", e)
        }

        val quoteVerifier = new SGXDCAPQuoteVerifierImpl()
        val verifyQuoteResult = quoteVerifier.verifyQuote(quote)
        if (verifyQuoteResult == 0) {
            System.out.println("Quote Verification Success!")
            System.exit(0)
        } else {
            System.out.println("Quote Verification Fail! Application killed")
            System.exit(1)
        }

    }

}
