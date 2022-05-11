package com.intel.analytics.bigdl.ppml.attestation

import com.intel.analytics.bigdl.ppml.kms.EHSMKeyManagementService
import org.apache.logging.log4j.LogManager
import scopt.OptionParser

object AttestationCLI {
    def main(args: Array[String]): Unit = {

        val logger = LogManager.getLogger(getClass)
        case class CmdParams(appID: String = "test",
                             appKey: String = "test",
                             asType: String = ATTESTATION_CONVENTION.MODE_EHSM_KMS,
                             asURL: String = "127.0.0.1",
                             userReport: String = "ppml")

        val cmdParser = new OptionParser[CmdParams]("PPML Attestation Quote Generation Cmd tool") {
            opt[String]('i', "appID")
              .text("app id for this app")
              .action((x, c) => c.copy(appID = x))
            opt[String]('k', "appKey")
              .text("app key for this app")
              .action((x, c) => c.copy(appKey = x))
            opt[String]('u', "asURL")
              .text("attestation service url, default is 127.0.0.1")
              .action((x, c) => c.copy(asURL = x))
            opt[String]('t', "asType")
              .text("attestation service type, default is EHSMKeyManagementService")
              .action((x, c) => c.copy(asURL = x))
            opt[String]('p', "userReport")
              .text("userReportDataPath, default is test")
              .action((x, c) => c.copy(userReport = x))

        }
        val params = cmdParser.parse(args, CmdParams()).get

        // Generate quote
        val userReportData = params.userReport
        val quoteGenerator = new QuoteGeneration()
        val quote = quoteGenerator.getQuote(userReportData)

        // Attestation Client
        val as = params.asType match {
            case ATTESTATION_CONVENTION.MODE_EHSM_KMS =>
                new EHSMAttestationService(params.asURL.split(":")(0),
                    params.asURL.split(":")(1), params.appID, params.appKey)
            case ATTESTATION_CONVENTION.MODE_DUMMY =>
                new DummyAttestationService()
            case _ => throw new AttestationRuntimeException("Wrong Attestation service type")
        }
        val attResult = as.attestWithServer(quote)
        // System.out.print(as.attestWithServer(quote))
        if (attResult._1) {
            System.out.println("Attestation Success!")
            // Bash success
            System.exit(0)
        } else {
            System.out.println("Attestation Fail! Application killed!")
            // bash fail
            System.exit(1)
        }
    }
}
