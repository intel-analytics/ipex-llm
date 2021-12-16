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


import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceGrpc;
import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.*;
import io.grpc.Channel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class FGBoostStub {
    private static final Logger logger = LoggerFactory.getLogger(NNStub.class);
    private static FGBoostServiceGrpc.FGBoostServiceBlockingStub stub;
    String clientID;
    public FGBoostStub(Channel channel, String clientID) {
        this.clientID = clientID;
        stub = FGBoostServiceGrpc.newBlockingStub(channel);
    }
    public UploadResponse uploadSplit(DataSplit ds) {
        UploadSplitRequest uploadRequest = UploadSplitRequest
                .newBuilder()
                .setSplit(ds)
                .setClientuuid(clientID)
                .build();

        return stub.uploadSplitTrain(uploadRequest);
    }

    /***
     * XGBoost download aggregated best split
     * @param treeID
     * @return
     */
    public DownloadSplitResponse downloadSplit(
            String treeID,
            String nodeID) {
        DownloadSplitRequest downloadRequest = DownloadSplitRequest
                .newBuilder()
                .setTreeID(treeID)
                .setNodeID(nodeID)
                .setClientuuid(clientID)
                .build();
        return stub.downloadSplitTrain(downloadRequest);
    }

    public UploadResponse uploadTreeEval(
            List<BoostEval> boostEval) {
        UploadTreeEvalRequest uploadTreeEvalRequest = UploadTreeEvalRequest
                .newBuilder()
                .setClientuuid(clientID)
                .addAllTreeEval(boostEval)
                .build();

        return stub.uploadTreeEval(uploadTreeEvalRequest);
    }

    public PredictTreeResponse uploadTreePred(
            List<BoostEval> boostEval) {
        PredictTreeRequest request = PredictTreeRequest
                .newBuilder()
                .setClientuuid(clientID)
                .addAllTreeEval(boostEval)
                .build();

        return stub.predictTree(request);
    }


    public UploadResponse uploadTreeLeaves(
            String treeID,
            List<Integer> treeIndexes,
            List<Float> treeOutput
    ) {
        TreeLeaves treeLeaves = TreeLeaves
                .newBuilder()
                .setTreeID(treeID)
                .addAllLeafIndex(treeIndexes)
                .addAllLeafOutput(treeOutput)
                .build();
        UploadTreeLeavesRequest uploadTreeLeavesRequest = UploadTreeLeavesRequest
                .newBuilder()
                .setClientuuid(clientID)
                .setTreeLeaves(treeLeaves)
                .build();
        return  stub.uploadTreeLeaves(uploadTreeLeavesRequest);
    }
}
