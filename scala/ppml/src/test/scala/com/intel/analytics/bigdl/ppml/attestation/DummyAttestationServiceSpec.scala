package com.intel.analytics.bigdl.ppml.attestation

import org.apache.logging.log4j.Logger
import org.scalatest.{FlatSpec, Matchers}

class DummyAttestationServiceSpec extends FlatSpec with Matchers {

    val dummyAttestationService = new DummyAttestationService()
    val logger: Logger = dummyAttestationService.logger

    "Get Quote " should "work" in {
        val quote = dummyAttestationService.getQuoteFromServer()
        quote shouldNot equal ("")
        logger.info(quote)
    }

    "Attest With Server " should "work" in {
        val trueResult = dummyAttestationService.attestWithServer("a_true_quote")._1
        trueResult should equal (true)
        val falseResult = dummyAttestationService.attestWithServer("a_false_quote")._1
        falseResult should equal (false)
    }
}
