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

import com.google.protobuf.ByteString;
import com.intel.analytics.bigdl.ckks.CKKS;
import com.intel.analytics.bigdl.ppml.fl.generated.FlBaseProto.*;
import com.intel.analytics.bigdl.ppml.fl.generated.NNServiceProto.*;
import com.intel.analytics.bigdl.ppml.fl.generated.NNServiceGrpc;
import io.grpc.Channel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class NNStub {
    private static final Logger logger = LoggerFactory.getLogger(NNStub.class);
    private static NNServiceGrpc.NNServiceBlockingStub stub;
    Integer clientID;
    protected CKKS ckks;
    protected long encrytorPtr;
    public NNStub(Channel channel, Integer clientID) {
        this.clientID = clientID;
        stub = NNServiceGrpc.newBlockingStub(channel);
    }

    public NNStub(Channel channel, Integer clientID, byte[][] secrets) {
        this.clientID = clientID;
        stub = NNServiceGrpc.newBlockingStub(channel);
        ckks = new CKKS();
        encrytorPtr = ckks.createCkksEncryptor(secrets);
    }

    private EncryptedTensor encrypt(FloatTensor ft) {
        float[] array = new float[ft.getTensorCount()];
        for(int i = 0; i < ft.getTensorCount(); i++) {
            array[i] = ft.getTensorList().get(i);
        }
        byte[] encryptedArray = ckks.ckksEncrypt(encrytorPtr, array);

        EncryptedTensor et =
          EncryptedTensor.newBuilder()
            .addAllShape(ft.getShapeList())
            .setTensor(ByteString.copyFrom(encryptedArray))
            .build();
        return et;
    }

    public TensorMap encrypt(TensorMap data) {
        Map<String, FloatTensor> d = data.getTensorMapMap();

        TensorMap.Builder encryptedDataBuilder = TensorMap.newBuilder()
          .setMetaData(data.getMetaData());
        for (Map.Entry<String, FloatTensor> fts: d.entrySet()){
            encryptedDataBuilder.putEncryptedTensorMap(fts.getKey(),
              encrypt(fts.getValue()));
        }
        return encryptedDataBuilder.build();
    }

    private FloatTensor decrypt(EncryptedTensor et) {
        byte[] array = et.getTensor().toByteArray();
        float[] decryptedArray = ckks.ckksDecrypt(encrytorPtr, array);

        List<Float> floatList = new ArrayList<Float>(decryptedArray.length);
        for (float v : decryptedArray) {
            floatList.add(v);
        }

        FloatTensor ft =
          FloatTensor.newBuilder()
            .addAllShape(et.getShapeList())
            .addAllTensor(floatList)
            .build();
        return ft;
    }

    public TensorMap decrypt(TensorMap data) {
        Map<String, EncryptedTensor> d = data.getEncryptedTensorMapMap();

        TensorMap.Builder encryptedDataBuilder = TensorMap.newBuilder()
          .setMetaData(data.getMetaData());
        for (Map.Entry<String, EncryptedTensor> ets: d.entrySet()){
            encryptedDataBuilder.putTensorMap(ets.getKey(),
              decrypt(ets.getValue()));
        }
        return encryptedDataBuilder.build();
    }

    public TensorMap train(TensorMap data, String algorithm) {
        TrainRequest.Builder trainRequestBuilder = TrainRequest
          .newBuilder()
          .setClientuuid(clientID)
          .setAlgorithm(algorithm);
        if (null != ckks) {
          TensorMap encryptedData = encrypt(data);
          trainRequestBuilder.setData(encryptedData);
          logDebugMessage(encryptedData);
          return decrypt(stub.train(trainRequestBuilder.build()).getData());
        } else {
          trainRequestBuilder.setData(data);
          logDebugMessage(data);
          return stub.train(trainRequestBuilder.build()).getData();
        }
    }


    public EvaluateResponse evaluate(TensorMap data, String algorithm, Boolean hasReturn) {
        EvaluateRequest.Builder evaluateRequestBuilder = EvaluateRequest
                .newBuilder()
                .setReturn(hasReturn)
                .setClientuuid(clientID)
                .setAlgorithm(algorithm);
        if (null != ckks) {
// TODO: evaluate with CKKS
//            TensorMap encryptedData = encrypt(data);
//            evaluateRequestBuilder.setData(encryptedData);
//            logDebugMessage(encryptedData);
            throw new UnsupportedOperationException("evaluate with CKKS is unspported.");
        } else {
            evaluateRequestBuilder.setData(data);
            logDebugMessage(data);
        }
        return stub.evaluate(evaluateRequestBuilder.build());
    }

    public TensorMap predict(TensorMap data, String algorithm) {
        PredictRequest.Builder predictRequestBuilder = PredictRequest
                .newBuilder()
                .setData(data)
                .setClientuuid(clientID)
                .setAlgorithm(algorithm);
        if (null != ckks) {
            TensorMap encryptedData = encrypt(data);
            predictRequestBuilder.setData(encryptedData);
            logDebugMessage(encryptedData);
            return decrypt(stub.predict(predictRequestBuilder.build()).getData());
        } else {
            predictRequestBuilder.setData(data);
            logDebugMessage(data);
            return stub.predict(predictRequestBuilder.build()).getData();
        }
    }

    private void logDebugMessage(TensorMap data) {
//        logger.debug("Upload the following data:");
//        logger.debug("Upload Data Name:" + data.getMetaData().getName());
//        logger.debug("Upload Data Version:" + data.getMetaData().getVersion());
//        logger.debug("Upload Data:");
//        logger.debug(data.getTensorMapMap().toString());
    }
}
