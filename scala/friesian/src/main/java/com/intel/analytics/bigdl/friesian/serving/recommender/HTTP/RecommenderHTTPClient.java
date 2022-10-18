package com.intel.analytics.bigdl.friesian.serving.recommender.HTTP;

import javax.ws.rs.client.Client;
import javax.ws.rs.client.ClientBuilder;
import javax.ws.rs.client.WebTarget;

public class RecommenderHTTPClient {
    public static void main(String[] args) {
        Client c = ClientBuilder.newClient();
        WebTarget target = c.target("http://10.239.158.177:8080/");
        for (int i = 1; i != 20; i ++) {
            String res = target.path("recommender/recommend/" + i).request().get(String.class);
            System.out.println(i + ": " + res);
        }
        System.out.println("--------------------metrics---------------------");
        System.out.println(target.path("recommender/metrics").request().get(String.class));
        System.out.println(target.path("recommender/clientMetrics").request().get(String.class));
        target.path("recommender/resetMetrics").request().get(String.class);
    }
}
