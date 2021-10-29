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

import com.intel.analytics.bigdl.ppml.generated.FLProto;
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
    public FLProto.DownloadResponse downloadTrain(String modelName, int flVersion) {
        logger.info("Download the following data:");
        FLProto.TableMetaData metadata = FLProto.TableMetaData.newBuilder()
                .setName(modelName).setVersion(flVersion + 1).build();
        FLProto.DownloadRequest downloadRequest = FLProto.DownloadRequest.newBuilder().setMetaData(metadata).build();
        return stub.downloadTrain(downloadRequest);
    }

    public FLProto.UploadResponse uploadTrain(FLProto.Table data) {

        FLProto.UploadRequest uploadRequest = FLProto.UploadRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .build();

        logger.info("Upload the following data:");
        logger.info("Upload Data Name:" + data.getMetaData().getName());
        logger.info("Upload Data Version:" + data.getMetaData().getVersion());
        logger.debug("Upload Data" + data.getTableMap());
//        logger.info("Upload" + data.getTableMap().get("weights").getTensorList().subList(0, 5));

        FLProto.UploadResponse uploadResponse = stub.uploadTrain(uploadRequest);
        return uploadResponse;
    }

    public FLProto.EvaluateResponse evaluate(FLProto.Table data, boolean lastBatch) {
        FLProto.EvaluateRequest eRequest = FLProto.EvaluateRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .setLast(lastBatch)
                .build();

        return stub.uploadEvaluate(eRequest);
    }
}
