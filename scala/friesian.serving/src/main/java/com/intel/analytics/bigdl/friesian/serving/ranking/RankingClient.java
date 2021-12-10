package com.intel.analytics.bigdl.friesian.serving.ranking;

import com.google.common.annotations.VisibleForTesting;
import com.google.protobuf.Message;
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity;
import com.intel.analytics.bigdl.dllib.utils.Table;
import com.intel.analytics.bigdl.friesian.serving.feature.utils.RedisUtils;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingProto.*;
import com.intel.analytics.bigdl.dllib.NNContext;
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils;
import com.intel.analytics.bigdl.friesian.serving.utils.Utils;
import io.grpc.Channel;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc.RankingBlockingStub;
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc.RankingStub;
import com.intel.analytics.bigdl.friesian.serving.utils.ranking.RankingUtils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Logger;
import org.apache.commons.cli.*;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;

public class RankingClient {
    private static final Logger logger = Logger.getLogger(RankingClient.class.getName());

    private final RankingBlockingStub blockingStub;
    private final RankingStub asyncStub;
    private TestHelper testHelper;

    /** Construct client for accessing AZInference server using the existing channel. */
    public RankingClient(Channel channel) {
        blockingStub = RankingGrpc.newBlockingStub(channel);
        asyncStub = RankingGrpc.newStub(channel);
    }

    public void inference(String encodedStr) {
//        info("*** Get input: " + jsonStr);

        Content request = Content.newBuilder().setEncodedStr(encodedStr).build();

        Prediction predResult;
        try {
            predResult = blockingStub.doPredict(request);
            if (testHelper != null) {
                testHelper.onMessage(predResult);
            }
        } catch (StatusRuntimeException e) {
            logger.warn("RPC failed: " + e.getStatus().toString());
            if (testHelper != null) {
                testHelper.onRpcError(e);
            }
            return;
        }
        Activity result = (Activity) EncodeUtils.bytesToObj(Base64.getDecoder().decode(predResult.getPredictStr()));
        System.out.println("Got predResult: " + RankingUtils.activityToList(result));
    }

    /** Issues several different requests and then exits. */
    public static void main(String[] args) throws InterruptedException, IOException, ParseException {
        Options options = new Options();
        Option target = new Option("t", "target", true, "The server to connect to.");
        options.addOption(target);
        Option threadNum = new Option("threadNum", true, "Thread number.");
        options.addOption(threadNum);

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
        int concurrentNum = Integer.parseInt(cmd.getOptionValue("threadNum", "1"));

        SparkContext sc = NNContext.initNNContext("WND inference");
        sc.setLogLevel("WARN");
        SparkSession spark = SparkSession.builder().config(sc.getConf()).getOrCreate();

        String dataDir = "/home/kai/test_wnd_inference/preprocessed_sample_train";
        Table input = RankingUtils.loadParquetAndConvert(spark, dataDir);
        String inputString = Base64.getEncoder().encodeToString(EncodeUtils.objToBytes(input));
        ManagedChannel channel = ManagedChannelBuilder.forTarget(targetURL).usePlaintext().build();
        try {
            ArrayList<InferenceThread> tList = new ArrayList<>();
            for (int i = 0; i < concurrentNum; i ++) {
                InferenceThread t = new InferenceThread(channel, inputString);
                tList.add(t);
                t.start();
            }
            for (InferenceThread t: tList) {
                t.join();
            }
        } finally {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

//    private void info(String msg, Object... params) {
//        logger.log(Level.INFO, msg, params);
//    }

//    private void warning(String msg, Object... params) {
//        logger.log(Level.WARNING, msg, params);
//    }

    /**
     * Only used for helping unit test.
     */
    @VisibleForTesting
    interface TestHelper {
        /**
         * Used for verify/inspect message received from server.
         */
        void onMessage(Message message);

        /**
         * Used for verify/inspect error received from server.
         */
        void onRpcError(Throwable exception);
    }

    @VisibleForTesting
    void setTestHelper(TestHelper testHelper) {
        this.testHelper = testHelper;
    }
}

class InferenceThread extends Thread {
    private ManagedChannel channel;
    private String msg;

    InferenceThread(ManagedChannel channel, String msg) {
        this.channel = channel;
        this.msg = msg;
    }

    @Override
    public void run() {
        RankingClient client = new RankingClient(channel);
        long start = System.nanoTime();
        for(int i = 0; i < 10; i ++) {
            client.inference(msg);
        }
        long end = System.nanoTime();
        long time = (end - start)/1000;
        System.out.println("time: " + time);
    }
}