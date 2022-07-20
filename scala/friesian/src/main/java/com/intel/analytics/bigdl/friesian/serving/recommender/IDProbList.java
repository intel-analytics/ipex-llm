package com.intel.analytics.bigdl.friesian.serving.recommender;

import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import com.fasterxml.jackson.annotation.JsonIgnore;

import java.util.Arrays;

public class IDProbList {
    private int[] ids;
    private float[] probs;
    private boolean success;
    private Status.Code errorCode;
    private String errorMsg;

    public IDProbList(int[] ids, float[] probs) {
        this.ids = ids;
        this.probs = probs;
        this.success = true;
    }

    public IDProbList(Status.Code errorCode, String errorMsg) {
        this.success = false;
        this.errorCode = errorCode;
        this.errorMsg = errorMsg;
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

    @JsonIgnore
    public StatusRuntimeException getgRPCException() {
        return this.errorCode.toStatus().withDescription("CandidateNum" +
                " should be larger than recommendNum.").asRuntimeException();
    }

    public String getErrorMsg() {
        return errorMsg;
    }

    public Status.Code getErrorCode() {
        return errorCode;
    }

    @Override
    public String toString() {
        return "IDProbList{" +
                "ids=" + Arrays.toString(ids) +
                ", probs=" + Arrays.toString(probs) +
                ", success=" + success +
//                ", gRPCException=" + gRPCException +
                '}';
    }
}
