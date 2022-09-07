import torch
from bigdl.nano.pytorch import Trainer
from bigdl.chronos.model.tcn import model_creator
from bigdl.chronos.metric.forecast_metrics import Evaluator
from bigdl.chronos.data.repo_dataset import get_public_dataset
from sklearn.preprocessing import StandardScaler

def gen_dataloader():
    tsdata_train, tsdata_val,\
        tsdata_test = get_public_dataset(name='nyc_taxi',
                                        with_split=True,
                                        val_ratio=0.1,
                                        test_ratio=0.1
                                        )
    # carry out additional customized preprocessing on the dataset.
    stand = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
            .impute()\
            .gen_dt_feature()\
            .scale(stand, fit=tsdata is tsdata_train)\
            .roll(lookback=48,horizon=1)

    tsdata_traindataloader = tsdata_train.to_torch_data_loader(batch_size=32)
    tsdata_valdataloader = tsdata_val.to_torch_data_loader(batch_size=32, shuffle=False)
    tsdata_testdataloader = tsdata_test.to_torch_data_loader(batch_size=32, shuffle=False)

    return tsdata_traindataloader,\
           tsdata_valdataloader,\
           tsdata_testdataloader

def predict_wraper(model, input_sample):
    model(input_sample)

if __name__ == '__main__':
    tsdata_traindataloader,\
    tsdata_valdataloader,\
    tsdata_testdataloader = gen_dataloader()
    
    config = {'input_feature_num':8,
              'output_feature_num':1,
              'past_seq_len':48,
              'future_seq_len':1,
              'kernel_size':3,
              'repo_initialization':True,
              'dropout':0.1,
              'seed': 0,
              'num_channels':[30]*7
              }
    model = model_creator(config)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    lit_model = Trainer.compile(model, loss, optimizer)
    trainer = Trainer(max_epochs=3, val_check_interval=1.0,
              accelerator='gpu', 
              devices=1,
              )
    trainer.fit(lit_model, tsdata_traindataloader, tsdata_testdataloader)
    
    x = None
    for x, _ in tsdata_traindataloader:
        break
    input_sample = x[0].unsqueeze(0)
    
    print("original pytorch latency (ms):", Evaluator.get_latency(predict_wraper, lit_model, input_sample))
    
    speed_model = Trainer.trace(lit_model, accelerator="onnxruntime", input_sample=input_sample)
    print("onnxruntime latency (ms):", Evaluator.get_latency(predict_wraper, speed_model, input_sample))
