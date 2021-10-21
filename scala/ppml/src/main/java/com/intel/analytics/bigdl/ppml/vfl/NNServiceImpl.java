package com.intel.analytics.bigdl.ppml.vfl;

import com.intel.analytics.bigdl.ppml.common.Aggregator;
import com.intel.analytics.bigdl.ppml.generated.FLProto.*;
import com.intel.analytics.bigdl.ppml.generated.NNServiceGrpc;
import io.grpc.stub.StreamObserver;

import static com.intel.analytics.bigdl.ppml.common.FLPhase.*;
import static com.intel.analytics.bigdl.ppml.common.FLPhase.TRAIN;

public class NNServiceImpl extends NNServiceGrpc.NNServiceImplBase {
    private Aggregator aggregator;

    public void setAggregator(Aggregator aggregator) {
        this.aggregator = aggregator;
    }

    @Override
    public void downloadTrain(
            DownloadRequest request, StreamObserver<DownloadResponse> responseObserver) {
        int version = request.getMetaData().getVersion();
        Table data = aggregator.getServerData(TRAIN).serverData;
        String response;
        if (data == null) {
            response = "Your required data doesn't exist";
            responseObserver.onNext(DownloadResponse.newBuilder().setResponse(response).setCode(0).build());
        } else {
            response = "Download data successfully";
            responseObserver.onNext(
                    DownloadResponse.newBuilder().setResponse(response).setData(data).setCode(1).build());
        }
        responseObserver.onCompleted();
    }

    @Override
    public void uploadTrain(
            UploadRequest request, StreamObserver<UploadResponse> responseObserver) {
        // check data version, drop all the unmatched version

        String clientUUID = request.getClientuuid();
        Table data = request.getData();
        int version = data.getMetaData().getVersion();

        try {
            aggregator.putClientData(TRAIN, clientUUID, version, data);
            UploadResponse response = UploadResponse.newBuilder().setResponse("Data received").setCode(0).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } catch (Exception e) {
            UploadResponse response = UploadResponse.newBuilder().setResponse(e.getMessage()).setCode(1).build();
            responseObserver.onNext(response);
            responseObserver.onCompleted();
        } finally {

        }
    }
}
