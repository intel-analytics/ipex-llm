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
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.*;
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
    public DownloadResponse downloadLabel(String modelName, int flVersion) {
        logger.info("Download the following data:");
        TableMetaData metadata = TableMetaData.newBuilder()
                .setName(modelName).setVersion(flVersion + 1).build();
        DownloadLabelRequest downloadRequest = DownloadLabelRequest.newBuilder().setMetaData(metadata).build();
        return stub.downloadLabel(downloadRequest);
    }

    public UploadResponse uploadLabel(Table data) {

        UploadLabelRequest uploadRequest = UploadLabelRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .build();

        logger.info("Upload the following data:");
        logger.info("Upload Data Name:" + data.getMetaData().getName());
        logger.info("Upload Data Version:" + data.getMetaData().getVersion());
        logger.debug("Upload Data" + data.getTableMap());
//        logger.info("Upload" + data.getTableMap().get("weights").getTensorList().subList(0, 5));

        UploadResponse uploadResponse = stub.uploadLabel(uploadRequest);
        return uploadResponse;
    }
    public SplitResponse split(DataSplit ds) {
        SplitRequest uploadRequest = SplitRequest
                .newBuilder()
                .setSplit(ds)
                .setClientuuid(clientID)
                .build();

        return stub.split(uploadRequest);
    }


    public EvaluateResponse evaluate(
            List<BoostEval> boostEval) {
        EvaluateRequest evaluateRequest = EvaluateRequest
                .newBuilder()
                .setClientuuid(clientID)
                .addAllTreeEval(boostEval)
                .build();

        return stub.evaluate(evaluateRequest);
    }

    public PredictResponse predict(
            List<BoostEval> boostEval) {
        PredictRequest request = PredictRequest
                .newBuilder()
                .setClientuuid(clientID)
                .addAllTreeEval(boostEval)
                .build();

        return stub.predict(request);
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
        return stub.uploadTreeLeaves(uploadTreeLeavesRequest);
    }
}
