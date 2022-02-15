/*
 * Copyright 2021 The BigDL Authors
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
import io.grpc.stub.StreamObserver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;


import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;

public class PSIServiceImpl extends PSIServiceGrpc.PSIServiceImplBase {
    private static final Logger logger = LogManager.getLogger(PSIServiceImpl.class);
    // TODO Thread safe
    protected PsiIntersection psiTask;
    // This psiCollections is
    //            TaskId,   ClientId, UploadRequest
    protected Map<String, String[]> psiCollections = new ConcurrentHashMap<>();
    int clientNum;
    String clientSalt;
    String clientSecret;
    // Stores the seed used in shuffling for each taskId
    int clientShuffleSeed = 0;
    protected int splitSize = 1000000;

    public PSIServiceImpl(int clientNum) {
        this.clientNum = clientNum;
    }
    @Override
    public void getSalt(SaltRequest req, StreamObserver<SaltReply> responseObserver) {
        String salt;
        // Store salt
        String taskId = req.getTaskId();
        if (clientSalt != null) {
            salt = clientSalt;
        } else {
            salt = Utils.getRandomUUID();
            clientSalt = salt;
        }

        // Store secure
        if (clientSecret == null) {
            clientSecret = req.getSecureCode();
        } else if (!clientSecret.equals(req.getSecureCode())) {
            // TODO Reply empty
            salt = "";
        }
        // Store random seed for shuffling
        if (clientShuffleSeed == 0) {
            clientShuffleSeed = Utils.getRandomInt();
        }
        SaltReply reply = SaltReply.newBuilder().setSaltReply(salt).build();
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }

    @Override
    public void uploadSet(UploadSetRequest request,
                          StreamObserver<UploadSetResponse> responseObserver) {
        SIGNAL signal;

        signal= SIGNAL.SUCCESS;
        String clientId = request.getClientId();
        int numSplit = request.getNumSplit();
        int splitLength = request.getSplitLength();
        int totalLength = request.getTotalLength();

        if(!psiCollections.containsKey(clientId)){
            if(psiCollections.size() >= clientNum) {
                logger.error("Too many clients, already has " +
                        psiCollections.keySet() +
                        ". The new one is " + clientId);
            }
            psiCollections.put(clientId, new String[totalLength]);
        }
        String[] collectionStorage = psiCollections.get(clientId);
        String[] ids = request.getHashedIDList().toArray(new String[request.getHashedIDList().size()]);
        int split = request.getSplit();
        // TODO: verify requests' splits are unique.
        System.arraycopy(ids, 0, collectionStorage, split * splitLength, ids.length);
        logger.info("ClientId" + clientId + ",split: " + split + ", numSplit: " + numSplit + ".");
        if (split == numSplit - 1) {
            synchronized (this) {
                try {
                    if (psiTask != null) {
                        logger.info("Adding " + (psiTask.numCollection() + 1) +
                                "th collections");
                        long st = System.currentTimeMillis();
                        psiTask.addCollection(collectionStorage);
                        logger.info("Added " + (psiTask.numCollection()) +
                                "th collections. Find Intersection time cost: " + (System.currentTimeMillis()-st) + " ms");
                    } else {
                        logger.info("Adding 1th collections.");
                        PsiIntersection pi = new PsiIntersection(clientNum,
                                clientShuffleSeed);
                        pi.addCollection(collectionStorage);
                        psiTask = pi;
                        logger.info("Added 1th collections.");
                    }
                    psiCollections.remove(clientId);
                } catch (InterruptedException | ExecutionException e){
                    logger.error(e.getMessage());
                    signal= SIGNAL.ERROR;
                } catch (IllegalArgumentException iae) {
                    logger.error("Current client ids are " + psiCollections.keySet());
                    logger.error(iae.getMessage());
                    throw iae;
                }
            }
        }


        UploadSetResponse response = UploadSetResponse.newBuilder()
                .setStatus(signal)
                .build();
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }

    @Override
    public void downloadIntersection(DownloadIntersectionRequest request,
                                     StreamObserver<DownloadIntersectionResponse> responseObserver) {
        SIGNAL signal = SIGNAL.SUCCESS;
        if (psiTask != null) {
            try {
                List<String> intersection = psiTask.getIntersection();
                if (intersection == null) {
                    DownloadIntersectionResponse response = DownloadIntersectionResponse.newBuilder()
                            .setStatus(SIGNAL.EMPTY_INPUT).build();
                    responseObserver.onNext(response);
                    responseObserver.onCompleted();
                    return;
                }
                int split = request.getSplit();
                int numSplit = Utils.getTotalSplitNum(intersection, splitSize);
                List<String> splitIntersection = Utils.getSplit(intersection, split, numSplit, splitSize);
                DownloadIntersectionResponse response = DownloadIntersectionResponse.newBuilder()
                        .setStatus(signal)
                        .setSplit(split)
                        .setNumSplit(numSplit)
                        .setTotalLength(intersection.size())
                        .setSplitLength(splitSize)
                        .addAllIntersection(splitIntersection).build();
                responseObserver.onNext(response);
                responseObserver.onCompleted();
            } catch (InterruptedException e) {
                logger.error(e.getMessage());
                signal = SIGNAL.ERROR;
                DownloadIntersectionResponse response = DownloadIntersectionResponse.newBuilder()
                        .setStatus(signal).build();
                responseObserver.onNext(response);
                responseObserver.onCompleted();
            }
        } else {
            DownloadIntersectionResponse response = DownloadIntersectionResponse.newBuilder()
                    .setStatus(SIGNAL.ERROR).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        }
    }


}

