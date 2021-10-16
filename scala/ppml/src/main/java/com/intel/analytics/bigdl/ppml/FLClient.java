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

package com.intel.analytics.bigdl.ppml;

import com.intel.analytics.bigdl.grpc.GrpcClientBase;
import com.intel.analytics.bigdl.ppml.generated.FLProto;
// import com.intel.analytics.bigdl.ppml.vfl.NNStub;
import com.intel.analytics.bigdl.ppml.psi.PSIStub;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class FLClient extends GrpcClientBase {
    private static final Logger logger = LoggerFactory.getLogger(FLClient.class);
    protected String taskID;
    /**
     * All supported FL implementations are listed below
     * VFL includes Private Set Intersection, Neural Network, Gradient Boosting
     */
    public PSIStub psiStub;
//    public NNStub nnStub;
    public FLClient() { this(null); }
    public FLClient(String[] args) {
        super(args);
    }

    @Override
    protected void parseConfig() throws IOException {
        FLHelper flHelper = getConfigFromYaml(FLHelper.class, configPath);
        if (flHelper != null) {
            target = flHelper.clientTarget;
            taskID = flHelper.taskID;
        }
        super.parseConfig();
    }

    @Override
    public void loadServices() {
        psiStub = new PSIStub(channel, taskID);
//        nnStub = new NNStub(channel, clientUUID);
    }

    public void shutdown() {
        try {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            logger.error("Shutdown Client Error" + e.getMessage());
        }
    }
    /**
     * Wrap all the api of stubs to expose the API out of the stubs
     */
    // PSI stub
    public String getSalt() {
        return psiStub.getSalt();
    }
    public String getSalt(String name, int clientNum, String secureCode) {
        return psiStub.getSalt(name, clientNum, secureCode);
    }
    public void uploadSet(List<String> hashedIdArray) {
        psiStub.uploadSet(hashedIdArray);
    }
    public List<String> downloadIntersection() {
        return psiStub.downloadIntersection();
    }
    // NN stub
//    public FLProto.DownloadResponse downloadTrain(String modelName, int flVersion) {
//        return nnStub.downloadTrain(modelName, flVersion);
//    }
//    public FLProto.UploadResponse uploadTrain(FLProto.Table data) {
//        return nnStub.uploadTrain(data);
//    }
//    public FLProto.EvaluateResponse evaluate(FLProto.Table data, boolean lastBatch) {
//        return nnStub.evaluate(data, lastBatch);
//    }


}
