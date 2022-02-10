package com.intel.analytics.bigdl.friesian.serving.feature.utils;

import io.lettuce.core.*;
import io.lettuce.core.api.StatefulRedisConnection;
import io.lettuce.core.api.async.RedisAsyncCommands;
import io.lettuce.core.api.sync.RedisStringCommands;
import io.lettuce.core.codec.StringCodec;
import io.lettuce.core.event.DefaultEventPublisherOptions;
import io.lettuce.core.event.EventBus;
import io.lettuce.core.event.metrics.CommandLatencyEvent;
import io.lettuce.core.masterreplica.MasterReplica;
import io.lettuce.core.masterreplica.StatefulRedisMasterReplicaConnection;
import io.lettuce.core.metrics.DefaultCommandLatencyCollector;
import io.lettuce.core.metrics.DefaultCommandLatencyCollectorOptions;
import io.lettuce.core.resource.ClientResources;
import io.lettuce.core.resource.DefaultClientResources;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import scala.Tuple2;

import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import com.intel.analytics.bigdl.friesian.serving.utils.Utils;

public class LettuceUtils {
    private static final Logger logger = LogManager.getLogger(LettuceUtils.class.getName());
    private static LettuceUtils instance = null;
    private static StatefulRedisMasterReplicaConnection<String, String> conn = null;
    // private static StatefulRedisConnection<String, String> conn2 = null;
    private static RedisClient redisClient;
    private String redisKeyPrefix;

    private LettuceUtils(ArrayList<Tuple2<String, Integer>> redisHostPort,
                       String redisPrefix) {
        int logInterval = 2;
        if (Utils.helper() != null) {
            logInterval = Utils.helper().logInterval();
        }
        ClientResources res = DefaultClientResources.builder()
                .commandLatencyPublisherOptions(
                        DefaultEventPublisherOptions.builder()
                                .eventEmitInterval(Duration.ofSeconds(2)).build())
                .commandLatencyRecorder(
                        new DefaultCommandLatencyCollector(
                                DefaultCommandLatencyCollectorOptions.builder()
                                        .targetPercentiles(new double[]{50.0, 95.0, 99.0}).build()))
                .build();
        redisClient = RedisClient.create(res);
        EventBus eventBus = redisClient.getResources().eventBus();
        eventBus.get()
                .filter(redisEvent -> redisEvent instanceof CommandLatencyEvent)
                .cast(CommandLatencyEvent.class)
                .subscribe(e -> logger.info(e.getLatencies()));
        List<RedisURI> nodes = new ArrayList<>(redisHostPort.size());
        for (int i = 0; i < redisHostPort.size(); i++) {
            nodes.add(RedisURI.create(redisHostPort.get(i)._1, redisHostPort.get(i)._2));
        }
        conn = MasterReplica.connect(redisClient, StringCodec.UTF8, nodes);
        conn.setReadFrom(ReadFrom.ANY_REPLICA);
        logger.info("Connected to Redis");
        redisKeyPrefix = redisPrefix;
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (conn != null) {
                conn.close();
            }
            if (redisClient != null) {
                redisClient.shutdown();
            }
        }));
    }

    public static LettuceUtils getInstance(ArrayList<Tuple2<String, Integer>> redisHostPort,
                                           String redisPrefix) {
        if (instance == null) {
            instance = new LettuceUtils(redisHostPort, redisPrefix);
        }
        return instance;
    }

    public RedisStringCommands<String, String> getSync() {
        return conn.sync();
    }

    public RedisAsyncCommands<String, String> getAsync() {
        return conn.async();
    }

    public String get(String key) {
        RedisAsyncCommands<String, String> async = getAsync();
        RedisFuture<String> future = async.get(key);
        String result = "";
        try {
            result = future.get(10000, TimeUnit.MILLISECONDS);
        } catch (InterruptedException | ExecutionException | TimeoutException e) {
            e.printStackTrace();
        }
        return result;
    }

    public void MSet(String keyPrefix, List<String>[] dataArray) {
        keyPrefix = redisKeyPrefix + keyPrefix;
        Map<String, String> keyValue = new HashMap<>();
        for(List<String> data: dataArray) {
            if(data.size() != 2) {
                logger.warn("Data size in dataArray should be 2, but got" + data.size());
            } else {
                keyValue.put(keyPrefix + ":" + data.get(0), data.get(1));
            }
        }
        RedisAsyncCommands<String, String> async = getAsync();
        async.mset(keyValue);
        logger.info(keyValue.size() + " valid records written to redis.");
    }

    public List<String> MGet(String keyPrefix, List<Integer> ids) {
        String[] keys = new String[ids.size()];
        for (int i = 0; i < ids.size(); i ++) {
            keys[i] = keyPrefix + ":" + ids.get(i);
        }
        RedisAsyncCommands<String, String> async = getAsync();
        RedisFuture<List<KeyValue<String, String>>> future = async.mget(keys);
        List<KeyValue<String, String>> result = null;
        try {
            result = future.get(10000, TimeUnit.MILLISECONDS);
        } catch (InterruptedException | ExecutionException | TimeoutException e) {
            e.printStackTrace();
        }
        List<String> values = new ArrayList<>(ids.size());
        if (result != null) {
            for (KeyValue<String, String> kv: result) {
                if (kv.hasValue()) {
                    values.add(kv.getValue());
                } else {
                    values.add("");
                }
            }
        }
        return values;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException, TimeoutException {
        ArrayList<Tuple2<String, Integer>> hostPort = new ArrayList<>();
//        hostPort.add(new Tuple2<>("localhost", 6379));
        hostPort.add(new Tuple2<>("10.239.158.177", 6380));
        hostPort.add(new Tuple2<>("localhost", 6381));
        LettuceUtils utils = LettuceUtils.getInstance(hostPort, "");
//        RedisStringCommands<String, String> sync = utils.getSync();

//        List<KeyValue<String, String>> value = sync.mget("a", "2tower_user", "d");
//        sync.set("b", "2");
        RedisAsyncCommands<String, String> async = utils.getAsync();
        Map<String, String> keyValue = new HashMap<>();
        keyValue.put("c", "c");
        keyValue.put("b", "b");
//        for (int  i = 1; i < 100000; i ++) {
////            RedisAsyncCommands<String, String> async = utils.getAsync();
//            async.mset(keyValue);
//        }
        for (int  i = 1; i < 100000; i ++) {
//            RedisAsyncCommands<String, String> async = utils.getAsync();
            RedisFuture<List<KeyValue<String, String>>> future = async.mget("a", "2tower_user", "b", "c");
            List<KeyValue<String, String>> result = future.get(1000, TimeUnit.MILLISECONDS);
            String v1 = result.get(2).getValue();
        }
        System.out.println("a");
    }
}
