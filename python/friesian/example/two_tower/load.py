import os
from bigdl.friesian.feature import FeatureTable
from bigdl.orca import init_orca_context, stop_orca_context

sc = init_orca_context()

# import tensorflow as tf

# model = tf.saved_model.load("recsys_wnd")

model_dir = "/home/yinchen/hamham223/BigDL/recsys_wnd"

wnd_item = FeatureTable.read_parquet(os.path.join(model_dir, "wnd_item.parquet"))
wnd_user = FeatureTable.read_parquet(os.path.join(model_dir, "wnd_user.parquet"))
item_embd = FeatureTable.read_parquet("item_ebd.parquet")
user_embd = FeatureTable.read_parquet("user_ebd.parquet")

print("wnd item size:" + str(wnd_item.size()))
print("wnd user size:" + str(wnd_user.size()))
print("item embd size:" + str(item_embd.size()))
print("user embd size:" + str(user_embd.size()))

stop_orca_context()
