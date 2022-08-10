# Choose proper forecasting model

How to choose a forecasting model among so many built-in models (or build one by yourself) in Chronos? That's a common question when users want to build their first forecasting model. Different forecasting models are more suitable for different data and different metrics(accuracy or performances).

The flowchart below is designed to guide our users which forecasting model to try on your own data. Click on the blocks in the chart below to see its documentation/examples.

```mermaid
flowchart TD
    StartPoint[I want to build a forecasting model]
    StartPoint-- always start from --> TCN[TCNForecaster]
    TCN -- performance is not satisfying --> TCN_OPT[Make sure optimizations are deploied]
    TCN_OPT -- further performance improvement is needed --> SER[Performance-awared Hyperparameter Optimization]
    SER -- only 1 step to be predicted --> LSTMForecaster
    SER -- only 1 var to be predicted --> NBeatsForecaster
    LSTMForecaster -- does not work --> CUS[customized model]
    NBeatsForecaster -- does not work --> CUS[customized model]

    TCN -- accuracy is not satisfying --> Tune[Hyperparameter Optimization]
    Tune -- only 1 step to be predicted --> LSTMForecaster2[LSTMForecaster]
    LSTMForecaster2 -- does not work --> AutoformerForecaster
    Tune -- more than 1 step to be predicted --> AutoformerForecaster
    AutoformerForecaster -- does not work --> Seq2SeqForecaster
    Seq2SeqForecaster -- does not work --> CUS[customized model]

    click TCN "https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#tcnforecaster"
    click LSTMForecaster "https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#lstmforecaster"
    click LSTMForecaster2 "https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#lstmforecaster"
    click NBeatsForecaster "https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#nbeatsforecaster"
    click Seq2SeqForecaster "https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#seq2seqforecaster"
    click AutoformerForecaster "."

    click StartPoint "https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/forecasting.html#seq2seqforecaster"
    click TCN_OPT "https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/speed_up.html"
    click SER "https://github.com/intel-analytics/BigDL/blob/main/python/chronos/example/hpo/muti_objective_hpo_with_builtin_latency_tutorial.ipynb"
    click Tune "https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/chronos-autotsest-quickstart.html"
    click CUS "https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/speed_up.html"

    classDef Model fill:#FFF,stroke:#0f29ba,stroke-width:1px;
    class TCN,LSTMForecaster,NBeatsForecaster,LSTMForecaster2,AutoformerForecaster,Seq2SeqForecaster Model;
```
