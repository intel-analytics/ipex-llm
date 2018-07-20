package com.intel.analytics.zoo.pipeline.inference;

public class InferenceRuntimeException extends RuntimeException {
	public InferenceRuntimeException(String msg) {
		super(msg);
	}

	public InferenceRuntimeException(String msg, Throwable cause) {
		super(msg, cause);
	}
}
