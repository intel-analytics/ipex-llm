package com.intel.analytics.bigdl.ppml.utils

import java.util.concurrent.TimeUnit

import com.intel.analytics.bigdl.ppml.FLContext

trait FLClientClosable {
  val flClient = FLContext.getClient()
  def close() = {
    flClient.getChannel.shutdownNow.awaitTermination(5, TimeUnit.SECONDS)
  }
}
