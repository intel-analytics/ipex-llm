/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.fl.vfl;


import com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceGrpc;
import com.intel.analytics.bigdl.ppml.fl.generated.FGBoostServiceProto.*;
import com.intel.analytics.bigdl.ppml.fl.generated.FlBaseProto.*;
import io.grpc.Channel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class FGBoostStub {
    private static final Logger logger = LoggerFactory.getLogger(FGBoostStub.class);
    private static FGBoostServiceGrpc.FGBoostServiceBlockingStub stub;
    Integer clientID;
    public FGBoostStub(Channel channel, Integer clientID) {
        this.clientID = clientID;
        stub = FGBoostServiceGrpc.newBlockingStub(channel);
    }
    public DownloadResponse downloadLabel(String modelName, int flVersion) {
//        logger.info("Download the following data:");
        MetaData metadata = MetaData.newBuilder()
                .setName(modelName).setVersion(flVersion + 1).build();
        DownloadLabelRequest downloadRequest = DownloadLabelRequest.newBuilder().setMetaData(metadata).build();
        return stub.downloadLabel(downloadRequest);
    }

    public UploadResponse uploadLabel(TensorMap data) {

        UploadLabelRequest uploadRequest = UploadLabelRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .build();

//        logger.info("Upload the following data:");
//        logger.info("Upload Data Name:" + data.getMetaData().getName());
//        logger.info("Upload Data Version:" + data.getMetaData().getVersion());
//        logger.debug("Upload Data" + data.getTensorMapMap());
//        logger.info("Upload" + data.getTensorMapMap().get("weights").getTensorList().subList(0, 5));

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
            List<BoostEval> boostEval, int version, boolean lastBatch) {
        EvaluateRequest evaluateRequest = EvaluateRequest
                .newBuilder()
                .setClientuuid(clientID)
                .addAllTreeEval(boostEval)
                .setVersion(version)
                .setLastBatch(lastBatch)
                .build();

        return stub.evaluate(evaluateRequest);
    }

    public PredictResponse predict(
            List<BoostEval> boostEval, int version) {
        PredictRequest request = PredictRequest
                .newBuilder()
                .setClientuuid(clientID)
                .addAllTreeEval(boostEval)
                .setVersion(version)
                .build();

        return stub.predict(request);
    }


    public UploadResponse uploadTreeLeaf(
            String treeID,
            List<Integer> treeIndexes,
            List<Float> treeOutput,
            int version
    ) {
        TreeLeaf treeLeaf = TreeLeaf
                .newBuilder()
                .setTreeID(treeID)
                .addAllLeafIndex(treeIndexes)
                .addAllLeafOutput(treeOutput)
                .setVersion(version)
                .build();
        UploadTreeLeafRequest uploadTreeLeafRequest = UploadTreeLeafRequest
                .newBuilder()
                .setClientuuid(clientID)
                .setTreeLeaf(treeLeaf)
                .build();
        return stub.uploadTreeLeaf(uploadTreeLeafRequest);
    }

    public SaveModelResponse saveServerModel(String modelPath) {
        SaveModelRequest saveModelRequest = SaveModelRequest
                .newBuilder()
                .setClientuuid(clientID)
                .setModelPath(modelPath)
                .build();
        return stub.saveServerModel(saveModelRequest);
    }

    public LoadModelResponse loadServerModel(String modelPath) {
        LoadModelRequest loadModelRequest = LoadModelRequest
                .newBuilder()
                .setClientId(clientID)
                .setModelPath(modelPath)
                .build();
        return stub.loadServerModel(loadModelRequest);
    }
}
