import os
import argparse
import json
import pandas as pd

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer

from models import *
from utils import VZArtifactCallback

NLP_ENCODER_PATH = "/home/kai/llm/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/7dbbc90392e2f80f3d3c277d6e90027e55de9125"


def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    else:
       return False

def load_params(params_path):
    with open(params_path, "r") as f:
        params = json.load(f)
    params_flatten = pd.json_normalize(params, sep=".").to_dict(orient="records")[0]
    return params, params_flatten


def main(params_path):
    params, params_flatten = load_params(params_path)
    
    OUTPUT_DIR = "./model_outputs/"
    DATA_VERSION = params["train_params"]["data_version"]
    LOOKUP_NAMES = params["train_params"]["lookup_names"]
    LOCAL_LOOKUP_DIRS = [
        f"/home/kai/vz-recommender-main/search_reco/product_ranking/sample/lookup/{lookup_name}/"
    for lookup_name in LOOKUP_NAMES
    ]
    
    trainer_device = 'cpu'
    gwdata = GridwallData(params, OUTPUT_DIR, "/home/kai/vz-recommender-main/search_reco/product_ranking/sample", LOCAL_LOOKUP_DIRS)
    gwdata.setup()
    tokenizer = AutoTokenizer.from_pretrained(NLP_ENCODER_PATH)
    params["hparam"]["loss_weights"] = gwdata.loss_weights
    torch_params = {
        **params["model"],
        "deep_dims": gwdata.deep_dims,
        "page_dim": gwdata.seq_num[0],
        "seq_dim": gwdata.seq_num[1],
        "item_meta_dim": gwdata.seq_num[2],
        "num_wide": int(len(params["train_params"]["wide_cols"])),
        "wad_embed_dim": params["model"]["wad_embed_dim"],
        "nlp_embed_dim": params["model"]["nlp_embed_dim"],
        "nlp_encoder_path": NLP_ENCODER_PATH,
        "task_out_dims": gwdata.task_out_dims,
    }
    pl_params = {
        "learning_rate": params["hparam"]["learning_rate"],
        "model_dir": "./out/",
        "hparam": params["hparam"],
        "k": 3,
        "log_interval":params["train_params"]["log_interval"],
        "experiment": params["train_params"]["experiment"],
        "model_kwargs": torch_params
    }
    
    params_list = [pl_params]
    for p in params_list:
        experiment = p["experiment"]
        artifact_dir = OUTPUT_DIR + "model_artifacts/" + str(experiment)
        model = GridwallPT(config=p, tokenizer=tokenizer)
        tb_logger = TensorBoardLogger(save_dir=OUTPUT_DIR, name=f"pzai_gridwall_{DATA_VERSION}_{experiment}", log_graph=False)
        earlystop_callback = EarlyStopping(monitor="loss/Validation", mode="min", min_delta=0.0, verbose=True, patience=0)
        artifact_callback = VZArtifactCallback(
            torch_model_params=p["model_kwargs"],
            model_dir=artifact_dir,
            params_to_pop=[p["model_kwargs"].pop("item_pre_embedding_weight", None)]
        )

        print("start training")
        trainer = Trainer(
                accelerator=trainer_device,
                devices=1,
                enable_checkpointing=False,
                max_epochs=params["hparam"]["num_epoch"],
                logger=tb_logger,
                log_every_n_steps=params["train_params"]["log_interval"],
                callbacks=[artifact_callback, earlystop_callback]
        )
        trainer.fit(model, gwdata)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params")
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    main(args.params)
