package com.intel.analytics.bigdl.apps.recommendation.inference;

public class UserItemPair {

    private final Integer userId;
    private final Integer itemId;

    public UserItemPair(Integer userId, Integer itemId){

        this.userId = userId;
        this.itemId = itemId;
    }

    public Integer getUserId() {
        return userId;
    }

    public Integer getItemId() {
        return itemId;
    }
}
