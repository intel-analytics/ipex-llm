package com.intel.analytics.bigdl.ppml.vfl.utils

import java.util.concurrent.TimeUnit

import com.intel.analytics.bigdl.ppml.FLClient
import com.intel.analytics.bigdl.ppml.vfl.VflContext

trait FLClientClosable {
  val flClient = VflContext.getClient()
  def close() = {
    flClient.getChannel.shutdownNow.awaitTermination(5, TimeUnit.SECONDS)
  }
}
