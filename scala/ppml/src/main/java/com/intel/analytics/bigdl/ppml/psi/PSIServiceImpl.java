/*
 * Copyright 2021 The Analytics Zoo Authors
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

import com.intel.analytics.bigdl.ppml.generated.FLProto.*;
import com.intel.analytics.bigdl.ppml.generated.PSIServiceGrpc;
import io.grpc.stub.StreamObserver;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;

public class PSIServiceImpl extends PSIServiceGrpc.PSIServiceImplBase {
    private static final Logger logger = LoggerFactory.getLogger(PSIServiceImpl.class);
    // TODO Thread safe
    protected Map<String, PsiIntersection> psiTasks = new ConcurrentHashMap<>();
    // This psiCollections is
    //            TaskId,   ClientId, UploadRequest
    protected Map<String, Map<String, String[]>> psiCollections = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, Integer> clientNum = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, String> clientSalt = new ConcurrentHashMap<>();
    ConcurrentHashMap<String, String> clientSecret = new ConcurrentHashMap<>();
    // Stores the seed used in shuffling for each taskId
    ConcurrentHashMap<String, Integer> clientShuffleSeed = new ConcurrentHashMap<>();
    protected int splitSize = 1000000;

    @Override
    public void getSalt(SaltRequest req, StreamObserver<SaltReply> responseObserver) {
        String salt;
        // Store salt
        String taskId = req.getTaskId();
        if (clientSalt.containsKey(taskId)) {
            salt = clientSalt.get(taskId);
        } else {
            salt = Utils.getRandomUUID();
            clientSalt.put(taskId, salt);
        }
        // Store clientNum
        if (req.getClientNum() != 0 && !clientNum.containsKey(taskId)) {
            clientNum.put(taskId, req.getClientNum());
        }
        // Store secure
        if (!clientSecret.containsKey(taskId)) {
            clientSecret.put(taskId, req.getSecureCode());
        } else if (!clientSecret.get(taskId).equals(req.getSecureCode())) {
            // TODO Reply empty
        }
        // Store random seed for shuffling
        if (!clientShuffleSeed.containsKey(taskId)) {
            clientShuffleSeed.put(taskId, Utils.getRandomInt());
        }
        SaltReply reply = SaltReply.newBuilder().setSaltReply(salt).build();
        responseObserver.onNext(reply);
        responseObserver.onCompleted();
    }

    @Override
    public void uploadSet(UploadSetRequest request,
                          StreamObserver<UploadSetResponse> responseObserver) {
        String taskId = request.getTaskId();
        SIGNAL signal;
        if (!clientNum.containsKey(taskId)) {
            signal= SIGNAL.ERROR;
            logger.error("TaskId not found in server, please get salt first. " +
                    "TaskID:" + taskId + ", ClientID:" + request.getClientId());
        } else {
            signal= SIGNAL.SUCCESS;
            String clientId = request.getClientId();
            int numSplit = request.getNumSplit();
            int splitLength = request.getSplitLength();
            int totalLength = request.getTotalLength();
            if(!psiCollections.containsKey(taskId)){
                psiCollections.put(taskId, new ConcurrentHashMap<String, String[]>());
            }
            if(!psiCollections.get(taskId).containsKey(clientId)){
                if(psiCollections.get(taskId).size() >= clientNum.get(taskId)) {
                    logger.error("Too many clients, already has " +
                            psiCollections.get(taskId).keySet() +
                            ". The new one is " + clientId);
                }
                psiCollections.get(taskId).put(clientId, new String[totalLength]);
            }
            String[] collectionStorage = psiCollections.get(taskId).get(clientId);
            String[] ids = request.getHashedIDList().toArray(new String[request.getHashedIDList().size()]);
            int split = request.getSplit();
            // TODO: verify requests' splits are unique.
            System.arraycopy(ids, 0, collectionStorage, split * splitLength, ids.length);
            logger.info("ClientId" + clientId + ",split: " + split + ", numSplit: " + numSplit + ".");
            if (split == numSplit - 1) {
                synchronized (psiTasks) {
                    try {
                        if (psiTasks.containsKey(taskId)) {
                            logger.info("Adding " + (psiTasks.get(taskId).numCollection() + 1) +
                                    "th collections to " + taskId + ".");
                            long st = System.currentTimeMillis();
                            psiTasks.get(taskId).addCollection(collectionStorage);
                            logger.info("Added " + (psiTasks.get(taskId).numCollection()) +
                                    "th collections to " + taskId + ". Find Intersection time cost: " + (System.currentTimeMillis()-st) + " ms");
                        } else {
                            logger.info("Adding 1th collections.");
                            PsiIntersection pi = new PsiIntersection(clientNum.get(taskId),
                                    clientShuffleSeed.get(taskId));
                            pi.addCollection(collectionStorage);
                            psiTasks.put(taskId, pi);
                            logger.info("Added 1th collections.");
                        }
                        psiCollections.get(taskId).remove(clientId);
                    } catch (InterruptedException | ExecutionException e){
                        logger.error(e.getMessage());
                        signal= SIGNAL.ERROR;
                    } catch (IllegalArgumentException iae) {
                        logger.error("TaskId " + taskId + ": Too many collections from client.");
                        logger.error("Current client ids are " + psiCollections.get(taskId).keySet());
                        logger.error(iae.getMessage());
                        throw iae;
                    }
                }
            }
        }

        UploadSetResponse response = UploadSetResponse.newBuilder()
                .setTaskId(taskId)
                .setStatus(signal)
                .build();
        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }

    @Override
    public void downloadIntersection(DownloadIntersectionRequest request,
                                     StreamObserver<DownloadIntersectionResponse> responseObserver) {
        String taskId = request.getTaskId();
        SIGNAL signal = SIGNAL.SUCCESS;
        if (psiTasks.containsKey(taskId)) {
            try {
                List<String> intersection = psiTasks.get(taskId).getIntersection();
                int split = request.getSplit();
                int numSplit = Utils.getTotalSplitNum(intersection, splitSize);
                List<String> splitIntersection = Utils.getSplit(intersection, split, numSplit, splitSize);
                DownloadIntersectionResponse response = DownloadIntersectionResponse.newBuilder()
                        .setTaskId(taskId)
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
                        .setTaskId(taskId)
                        .setStatus(signal).build();
                responseObserver.onNext(response);
                responseObserver.onCompleted();
            }
        }
    }


}

