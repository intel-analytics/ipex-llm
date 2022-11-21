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
import java.io.{File, FileInputStream, FileOutputStream}
import scopt.OptionParser

import com.intel.analytics.bigdl.ppml.attestation._

object QuoteGeneratorCmd {
    def main(args: Array[String]): Unit = {

        val logger = LogManager.getLogger(getClass)
        case class CmdParams(
            libOSType: String = "gramine",
            reportDataPath: String = "./userReportData",
            quoteOutputPath: String = "./quoteOutputDump"
        )

        val cmdParser = new OptionParser[CmdParams]("PPML Attestation Quote Generation Cmd tool") {
            opt[String]('l', "libos")
            .text("libOSType, default is gramine")
            .action((x, c) => c.copy(libOSType = x))
            opt[String]('r', "reportdata")
            .text("userReportDataPath, default is ./userReportData ")
            .action((x, c) => c.copy(reportDataPath = x))
            opt[String]('q', "quote")
            .text("quoteOutputPath, default is ./quoteOutputDump")
            .action((x, c) => c.copy(quoteOutputPath = x))
        }
        val params = cmdParser.parse(args, CmdParams()).get

        if(params.libOSType=="gramine") {
            val userReportData = try {
                // read userReportData
                val userReportDataFile = new File(params.reportDataPath)
                val in = new FileInputStream(userReportDataFile)
                val userReportData = new Array[Byte](userReportDataFile.length.toInt)
                in.read(userReportData)
                in.close()
                userReportData
            } catch {
                case e: Exception =>
                    logger.error("Failed to read user report data.")
                    throw new AttestationRuntimeException("Failed to read user report data " +
                    "from file to buffer!", e)
            }

            val gramineQuoteGenerator = new GramineQuoteGeneratorImpl()
            val quote = gramineQuoteGenerator.getQuote(userReportData)

            try {
                // write gramine-quote to file.
                val quoteWriter = new FileOutputStream(params.quoteOutputPath)
                quoteWriter.write(quote)
                quoteWriter.close()
            } catch {
            case e: Exception =>
                logger.error(s"Failed to write gramine quote to file, ${e}")
                throw new AttestationRuntimeException("Failed " +
                "to write gramine quote into a file!", e)
            }
        }
        else if (params.libOSType=="occlum") {
            // TODO
            logger.info("TODO libOS Type!")
            logger.info(params.reportDataPath)
            logger.info(params.quoteOutputPath)
        }
        else {

            logger.error("Unknown libOS Type!")

        }
    }
}
