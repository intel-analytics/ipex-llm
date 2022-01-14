/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.vfl;

import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.*;
import com.intel.analytics.bigdl.ppml.generated.NNServiceProto.*;
import com.intel.analytics.bigdl.ppml.generated.NNServiceGrpc;
import io.grpc.Channel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NNStub {
    private static final Logger logger = LoggerFactory.getLogger(NNStub.class);
    private static NNServiceGrpc.NNServiceBlockingStub stub;
    String clientID;
    public NNStub(Channel channel, String clientID) {
        this.clientID = clientID;
        stub = NNServiceGrpc.newBlockingStub(channel);
    }

    public TrainResponse train(TensorMap data, String algorithm) {
        TrainRequest trainRequest = TrainRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .setAlgorithm(algorithm)
                .build();
        logDebugMessage(data);
        return stub.train(trainRequest);
    }


    public EvaluateResponse evaluate(TensorMap data, String algorithm, Boolean hasReturn) {
        EvaluateRequest evaluateRequest = EvaluateRequest
                .newBuilder()
                .setData(data)
                .setReturn(hasReturn)
                .setClientuuid(clientID)
                .setAlgorithm(algorithm)
                .build();
        logDebugMessage(data);
        return stub.evaluate(evaluateRequest);
    }

    public PredictResponse predict(TensorMap data, String algorithm) {
        PredictRequest predictRequest = PredictRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .setAlgorithm(algorithm)
                .build();
        logDebugMessage(data);
        return stub.predict(predictRequest);
    }

    private void logDebugMessage(TensorMap data) {
        logger.debug("Upload the following data:");
        logger.debug("Upload Data Name:" + data.getMetaData().getName());
        logger.debug("Upload Data Version:" + data.getMetaData().getVersion());
        logger.debug("Upload Data:");
        logger.debug(data.getTensorsMap().toString());
    }
}
