package com.intel.analytics.bigdl.ppml.attestation

import org.scalatest.{FlatSpec, Matchers}

class DummyAttestationServiceSpec extends FlatSpec with Matchers {
    "Dummy Attestation " should "work" in {
        val dummyAttestationService = new DummyAttestationService()
        val logger = dummyAttestationService.logger
        val quote = dummyAttestationService.getQuoteFromServer()
        val result = dummyAttestationService.attestWithServer(quote)
        val verifyQuoteResult = result._1
        val response = result._2
        logger.info(quote)
        logger.info(response)
        logger.info(verifyQuoteResult)
    }
}
