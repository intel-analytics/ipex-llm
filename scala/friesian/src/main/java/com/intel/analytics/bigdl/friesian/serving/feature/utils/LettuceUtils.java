package com.intel.analytics.bigdl.friesian.serving.feature.utils;

import io.lettuce.core.*;
import io.lettuce.core.api.async.RedisAsyncCommands;
import io.lettuce.core.api.sync.RedisStringCommands;
import io.lettuce.core.cluster.ClusterClientOptions;
import io.lettuce.core.cluster.ClusterTopologyRefreshOptions;
import io.lettuce.core.cluster.RedisClusterClient;
import io.lettuce.core.cluster.api.StatefulRedisClusterConnection;
import io.lettuce.core.cluster.api.sync.RedisAdvancedClusterCommands;
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
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import com.intel.analytics.bigdl.friesian.serving.utils.Utils;

enum RedisType {
    STANDALONE, SENTINEL, CLUSTER
}

public class LettuceUtils {
    private static final Logger logger = LogManager.getLogger(LettuceUtils.class.getName());
    private static LettuceUtils instance = null;
    private static StatefulRedisMasterReplicaConnection<String, String> standaloneConn = null;
    private static StatefulRedisClusterConnection<String, String> clusterConn = null;
    private static AbstractRedisClient redisClient;
    private String redisKeyPrefix;
    private RedisType redisType;

    private LettuceUtils(ArrayList<Tuple2<String, Integer>> redisHostPort,
                       String redisPrefix, String sentinelMasterUrl, String sentinelMasterName) {
        int logInterval = 2;
        redisType = RedisType.STANDALONE;
        if (Utils.helper() != null) {
            logInterval = Utils.helper().logInterval();
            if (Utils.helper().redisType().equals("sentinel")) {
                redisType = RedisType.SENTINEL;
            } else if (Utils.helper().redisType().equals("cluster")) {
                redisType = RedisType.CLUSTER;
            }
        }
        ClientResources res;
        if (logInterval > 0) {
            res = DefaultClientResources.builder()
                    .commandLatencyPublisherOptions(
                            DefaultEventPublisherOptions.builder()
                                    .eventEmitInterval(Duration.ofMinutes(logInterval)).build())
                    .commandLatencyRecorder(
                            new DefaultCommandLatencyCollector(
                                    DefaultCommandLatencyCollectorOptions.builder()
                                            .targetPercentiles(new double[]{50.0, 95.0, 99.0}).build()))
                    .build();
        } else {
            res = DefaultClientResources.builder().build();
        }

        List<RedisURI> nodes = new ArrayList<>(redisHostPort.size());
        if (redisType == RedisType.SENTINEL || redisType == RedisType.STANDALONE) {
            RedisClient redisStandaloneClient = RedisClient.create(res);
            redisClient = redisStandaloneClient;
            if (redisType == RedisType.STANDALONE) {
                for (Tuple2<String, Integer> stringIntegerTuple2 : redisHostPort) {
                    nodes.add(RedisURI.create(stringIntegerTuple2._1, stringIntegerTuple2._2));
                }
            } else {
                String[] sentinelUrl = sentinelMasterUrl.split(":");
                assert sentinelUrl.length == 2;
                nodes.add(RedisURI.Builder.sentinel(sentinelUrl[0].trim(), Integer.parseInt(sentinelUrl[1].trim()),
                        sentinelMasterName).build());
            }
            standaloneConn = MasterReplica.connect(redisStandaloneClient, StringCodec.UTF8, nodes);
            if (redisHostPort.size() == 1) {
                standaloneConn.setReadFrom(ReadFrom.ANY);
            } else {
                standaloneConn.setReadFrom(ReadFrom.ANY_REPLICA);
            }
        } else {
            for (Tuple2<String, Integer> stringIntegerTuple2 : redisHostPort) {
                nodes.add(RedisURI.create(stringIntegerTuple2._1, stringIntegerTuple2._2));
            }
            RedisClusterClient redisClusterClient = RedisClusterClient.create(nodes);
            ClusterTopologyRefreshOptions topologyRefreshOptions =
                    ClusterTopologyRefreshOptions.builder()
                            .enableAdaptiveRefreshTrigger(ClusterTopologyRefreshOptions.RefreshTrigger.MOVED_REDIRECT,
                                    ClusterTopologyRefreshOptions.RefreshTrigger.PERSISTENT_RECONNECTS)
                            .adaptiveRefreshTriggersTimeout(Duration.ofSeconds(30)).build();
            redisClusterClient.setOptions(ClusterClientOptions.builder()
                            .topologyRefreshOptions(topologyRefreshOptions).build());
            clusterConn = redisClusterClient.connect();
        }

        logger.info("Connected to Redis");
        redisKeyPrefix = redisPrefix;

        if (logInterval > 0) {
            EventBus eventBus = redisClient.getResources().eventBus();
            eventBus.get()
                    .filter(redisEvent -> redisEvent instanceof CommandLatencyEvent)
                    .cast(CommandLatencyEvent.class)
                    .subscribe(e -> logger.info(e.getLatencies()));
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (standaloneConn != null) {
                standaloneConn.close();
            }
            if (redisClient != null) {
                redisClient.shutdown();
            }
        }));
    }

    public static LettuceUtils getInstance(ArrayList<Tuple2<String, Integer>> redisHostPort,
                                           String redisPrefix) {
        if (instance == null) {
            instance = new LettuceUtils(redisHostPort, redisPrefix, null, null);
        }
        return instance;
    }

    public static LettuceUtils getInstance(String redisPrefix, String sentinelMasterURL, String sentinelMasterName) {
        if (instance == null) {
            instance = new LettuceUtils(null, redisPrefix, sentinelMasterURL, sentinelMasterName);
        }
        return instance;
    }

    public RedisStringCommands<String, String> getSync() {
        if (redisType == RedisType.CLUSTER) {
            return clusterConn.sync();
        } else {
            return standaloneConn.sync();
        }
    }

    public RedisAsyncCommands<String, String> getAsync() {
        if (redisType == RedisType.CLUSTER) {
            // TODO: fix
            return (RedisAsyncCommands<String, String>) clusterConn.async();
        } else {
            return standaloneConn.async();
        }
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
        hostPort.add(new Tuple2<>("localhost", 6379));
//        hostPort.add(new Tuple2<>("10.239.158.177", 6380));
//        hostPort.add(new Tuple2<>("localhost", 6381));
        LettuceUtils utils = LettuceUtils.getInstance(hostPort, "");
//        RedisStringCommands<String, String> sync = utils.getSync();

//        List<KeyValue<String, String>> value = sync.mget("a", "2tower_user", "d");
//        sync.set("b", "2");
        RedisAsyncCommands<String, String> async = utils.getAsync();
        Map<String, String> keyValue = new HashMap<>();
        keyValue.put("c", "c");
        keyValue.put("b", "b");
        for (int  i = 1; i < 100000; i ++) {
//            RedisAsyncCommands<String, String> async = utils.getAsync();
            async.mset(keyValue);
        }
        for (int  i = 1; i < 100000; i ++) {
//            RedisAsyncCommands<String, String> async = utils.getAsync();
            RedisFuture<List<KeyValue<String, String>>> future = async.mget("a", "2tower_user", "b", "c");
            List<KeyValue<String, String>> result = future.get(1000, TimeUnit.MILLISECONDS);
            String v1 = result.get(2).getValue();
        }
        System.out.println("a");
    }
}
