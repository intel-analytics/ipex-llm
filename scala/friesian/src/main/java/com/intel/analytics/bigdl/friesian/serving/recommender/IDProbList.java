package com.intel.analytics.bigdl.friesian.serving.recommender;

import io.grpc.StatusRuntimeException;

public class IDProbList {
    private int[] ids;
    private float[] probs;
    private boolean success;
    private StatusRuntimeException gRPCException;

    public IDProbList(int[] ids, float[] probs) {
        this.ids = ids;
        this.probs = probs;
        this.success = true;
    }

    public IDProbList(StatusRuntimeException gRPCException) {
        this.success = false;
        this.gRPCException = gRPCException;
    }

    public int[] getIds() {
        return ids;
    }

    public float[] getProbs() {
        return probs;
    }

    public boolean isSuccess() {
        return success;
    }

    public StatusRuntimeException getgRPCException() {
        return gRPCException;
    }
}
