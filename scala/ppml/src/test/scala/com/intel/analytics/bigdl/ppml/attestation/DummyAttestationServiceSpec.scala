package com.intel.analytics.bigdl.ppml.attestation

import org.apache.logging.log4j.Logger
import org.scalatest.{FlatSpec, Matchers}

class DummyAttestationServiceSpec extends FlatSpec with Matchers {

    val dummyAttestationService = new DummyAttestationService()
    val logger: Logger = dummyAttestationService.logger
    var quote: String = ""

    "Get Quote " should "work" in {
        quote = dummyAttestationService.getQuoteFromServer()
        quote shouldNot equal ("")
        logger.info(quote)
    }

    "Attest With Server " should "work" in {
        var result = dummyAttestationService.attestWithServer(quote)._1
        val quoteResult = quote.indexOf("true") >= 0
        result should equal (quoteResult)
        result = dummyAttestationService.attestWithServer("a_true_quote")._1
        result should equal (true)
        result = dummyAttestationService.attestWithServer("a_test_quote")._1
        result should equal (false)
    }
}
