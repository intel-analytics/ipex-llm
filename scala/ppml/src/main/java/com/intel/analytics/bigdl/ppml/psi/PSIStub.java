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

package com.intel.analytics.bigdl.ppml.psi;

import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.*;
import com.intel.analytics.bigdl.ppml.generated.PSIServiceGrpc;
import com.intel.analytics.bigdl.ppml.generated.PSIServiceProto.*;

import io.grpc.Channel;
import io.grpc.StatusRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class PSIStub {
    private static final Logger logger = LoggerFactory.getLogger(PSIStub.class);
    private PSIServiceGrpc.PSIServiceBlockingStub stub;
    public PSIStub(Channel channel) {
        stub = PSIServiceGrpc.newBlockingStub(channel);
    }
    protected String clientID = UUID.randomUUID().toString();
    protected String salt;
    protected int splitSize = 1000000;

    public String getSalt() {
        return getSalt("");
    }
    /**
     * For PSI usage only
     * To get salt from FL Server, will get a new one if its salt does not exist on server
     * @param secureCode String, secure code
     * @return String, the salt get from server
     */
    public String getSalt(String secureCode) {
        logger.info(clientID + " getting salt from PSI service");
        SaltRequest request = SaltRequest.newBuilder()
                .setSecureCode(secureCode).build();
        SaltReply response;
        try {
            response = stub.getSalt(request);
        } catch (StatusRuntimeException e) {
            throw new RuntimeException("RPC failed: " + e.getMessage());
        }
        if (!response.getSaltReply().isEmpty()) {
            salt = response.getSaltReply();
        }
        return response.getSaltReply();
    }

    /**
     * For PSI usage only
     * Upload local set to FL Server in VFL
     * @param hashedIdArray List of String, the set trained at local
     */
    public void uploadSet(List<String> hashedIdArray) {
        int numSplit = Utils.getTotalSplitNum(hashedIdArray, splitSize);
        int split = 0;
        while (split < numSplit) {
            List<String> splitArray = Utils.getSplit(hashedIdArray, split, numSplit, splitSize);
            UploadSetRequest request = UploadSetRequest.newBuilder()
                    .setSplit(split)
                    .setNumSplit(numSplit)
                    .setSplitLength(splitSize)
                    .setTotalLength(hashedIdArray.size())
                    .setClientId(clientID)
                    .addAllHashedID(splitArray)
                    .build();
            try {
                stub.uploadSet(request);
            } catch (StatusRuntimeException e) {
                throw new RuntimeException("RPC failed: " + e.getMessage());
            }
            split ++;
        }
    }

    /**
     * For PSI usage only
     * Download intersection from FL Server in VFL
     * @return List of String, the intersection downloaded
     */
    public List<String> downloadIntersection() throws Exception {
        List<String> result = new ArrayList<String>();
        try {
            logger.info("Downloading 0th intersection");
            DownloadIntersectionRequest request = DownloadIntersectionRequest.newBuilder()
                    .setSplit(0)
                    .build();
            DownloadIntersectionResponse response = stub.downloadIntersection(request);
            if (response.getStatus() == SIGNAL.ERROR) {
                throw new Exception("Task ID does not exist on server, please upload set first.");
            }
            if (response.getStatus() == SIGNAL.EMPTY_INPUT) {
                // empty intersection, just return
                return null;
            }
            logger.info("Downloaded 0th intersection");
            result.addAll(response.getIntersectionList());
            for (int i = 1; i < response.getNumSplit(); i++) {
                request = DownloadIntersectionRequest.newBuilder()
                        .setSplit(i)
                        .build();
                logger.info("Downloading " + i + "th intersection");
                response = stub.downloadIntersection(request);
                logger.info("Downloaded " + i + "th intersection");
                result.addAll(response.getIntersectionList());
            }
            assert(result.size() == response.getTotalLength());
        } catch (StatusRuntimeException e) {
            throw new RuntimeException("RPC failed: " + e.getMessage());
        }
        return result;
    }
}
