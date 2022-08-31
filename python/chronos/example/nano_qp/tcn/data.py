from tcn_config import *
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
            .roll(lookback=lookback,horizon=horizon)

    tsdata_traindataloader = tsdata_train.to_torch_data_loader(batch_size=batch_size)
    tsdata_valdataloader = tsdata_val.to_torch_data_loader(batch_size=batch_size, shuffle=False)
    tsdata_testdataloader = tsdata_test.to_torch_data_loader(batch_size=batch_size, shuffle=False)

    return tsdata_traindataloader,\
           tsdata_valdataloader,\
           tsdata_testdataloader,\
           tsdata_test 