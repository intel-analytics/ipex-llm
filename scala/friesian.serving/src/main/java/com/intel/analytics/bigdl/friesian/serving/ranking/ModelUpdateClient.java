package com.intel.analytics.bigdl.friesian.serving.ranking;

import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc.RankingBlockingStub;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc.RankingStub;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingProto.ModelMeta;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingProto.Status;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.apache.commons.cli.*;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class ModelUpdateClient {
    private static final Logger logger = Logger.getLogger(RankingClient.class.getName());

    private final RankingBlockingStub blockingStub;
    private final RankingStub asyncStub;

    /** Construct client for accessing the RankingServer using the existing channel. */
    public ModelUpdateClient(Channel channel) {
        blockingStub = RankingGrpc.newBlockingStub(channel);
        asyncStub = RankingGrpc.newStub(channel);
    }

    public void addModel(String path, String version) {
        ModelMeta request = ModelMeta.newBuilder().setPath(path).setVersion(version).build();
        Status result;
        try {
            result = blockingStub.addModel(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            return;
        }
        System.out.println(result.getSuccess());
    }

    public void registerModel(String path, String version) {
        ModelMeta request = ModelMeta.newBuilder().setPath(path).setVersion(version).build();
        Status result;
        try {
            result = blockingStub.registerModel(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            return;
        }
        System.out.println(result.getSuccess());
    }

    public void deregisterModel(String path, String version) {
        ModelMeta request = ModelMeta.newBuilder().setPath(path).setVersion(version).build();
        Status result;
        try {
            result = blockingStub.deregisterModel(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            return;
        }
        System.out.println(result.getSuccess());
    }

    public void removeModel(String path, String version) {
        ModelMeta request = ModelMeta.newBuilder().setPath(path).setVersion(version).build();
        Status result;
        try {
            result = blockingStub.removeModel(request);
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            return;
        }
        System.out.println(result.getSuccess());
    }

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException, IOException, ParseException {
        Options options = new Options();
        Option target = new Option("t", "target", true, "The server to connect to.");
        options.addOption(target);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("utility-name", options);
            System.exit(1);
        }
        assert cmd != null;
        String targetURL = cmd.getOptionValue("target", "localhost:8083");
        String oldModel = "/home/kai/test_wnd_inference/recsys_wnd";
        String newModel = "/home/kai/test_wnd_inference/recsys_wnd_new";
        ManagedChannel channel = ManagedChannelBuilder.forTarget(targetURL).usePlaintext().build();
        ModelUpdateClient client = new ModelUpdateClient(channel);
        try {
            client.addModel(newModel, "v2");
            Thread.sleep(10000);  // leave some time gap for all services to load the new model
            client.addModel(newModel, "v2");  // adding the same model would do nothing
            Thread.sleep(50000);
            client.registerModel(newModel, "v2");  // the new model is now online
            Thread.sleep(50000);  // registering the same model would do nothing
            client.deregisterModel(oldModel, "v1");
            Thread.sleep(10000);  // leave some time gap for all services to offline the old model
            client.deregisterModel(oldModel, "v1");  // deregistering the same model would do nothing
            Thread.sleep(50000);
            client.removeModel(oldModel, "v1");
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }
}
