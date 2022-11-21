#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from bigdl.nano.utils.log4Error import invalidInputError


def generate_forecaster(args):
    input_feature_num = 321 if args.dataset == "tsinghua_electricity" else 1
    output_feature_num = 321 if args.dataset == "tsinghua_electricity" else 1
    metrics = args.metrics
    freq = 'h' if args.dataset == "tsinghua_electricity" else 't'

    if args.model == 'lstm':
        if args.framework == 'torch':
            from bigdl.chronos.forecaster import LSTMForecaster as LSTMForecaster_torch
            return LSTMForecaster_torch(past_seq_len=args.lookback,
                                        input_feature_num=input_feature_num,
                                        output_feature_num=output_feature_num,
                                        metrics=metrics)
        elif args.framework == 'tensorflow':
            from bigdl.chronos.forecaster.tf import LSTMForecaster as LSTMForecaster_tf
            return LSTMForecaster_tf(past_seq_len=args.lookback,
                                     input_feature_num=input_feature_num,
                                     output_feature_num=output_feature_num,
                                     metrics=metrics)
    elif args.model == 'tcn':
        if args.framework == 'torch':
            from bigdl.chronos.forecaster import TCNForecaster as TCNForecaster_torch
            return TCNForecaster_torch(past_seq_len=args.lookback,
                                       future_seq_len=args.horizon,
                                       input_feature_num=input_feature_num,
                                       output_feature_num=output_feature_num,
                                       normalization=args.normalization,
                                       decomposition_kernel_size=0,
                                       metrics=metrics)
        elif args.framework == 'tensorflow':
            from bigdl.chronos.forecaster.tf import TCNForecaster as TCNForecaster_tf
            return TCNForecaster_tf(past_seq_len=args.lookback,
                                    future_seq_len=args.horizon,
                                    input_feature_num=input_feature_num,
                                    output_feature_num=output_feature_num,
                                    metrics=metrics)

    elif args.model == 'seq2seq':
        if args.framework == 'torch':
            from bigdl.chronos.forecaster import Seq2SeqForecaster as Seq2SeqForecaster_torch
            return Seq2SeqForecaster_torch(past_seq_len=args.lookback,
                                           future_seq_len=args.horizon,
                                           input_feature_num=input_feature_num,
                                           output_feature_num=output_feature_num,
                                           metrics=metrics)
        elif args.framework == 'tensorflow':
            from bigdl.chronos.forecaster.tf import Seq2SeqForecaster as Seq2SeqForecaster_tf
            return Seq2SeqForecaster_tf(past_seq_len=args.lookback,
                                        future_seq_len=args.horizon,
                                        input_feature_num=input_feature_num,
                                        output_feature_num=output_feature_num,
                                        metrics=metrics)

    elif args.model == 'autoformer':
        if args.framework == 'torch':
            from bigdl.chronos.forecaster import AutoformerForecaster as AutoformerForecaster_torch
            return AutoformerForecaster_torch(past_seq_len=args.lookback,
                                              future_seq_len=args.horizon,
                                              input_feature_num=input_feature_num,
                                              output_feature_num=output_feature_num,
                                              label_len=int(args.lookback/2),
                                              freq=freq,
                                              metrics=metrics)
        else:
            invalidInputError(args.framework == 'torch',
                              f"Autoformer does not support tensorflow backend now.")

    elif args.model == 'nbeats':
        if args.framework == 'torch':
            from bigdl.chronos.forecaster import NBeatsForecaster as NBeatsForecaster_torch
            return NBeatsForecaster_torch(past_seq_len=args.lookback,
                                          future_seq_len=args.horizon,
                                          metrics=metrics)
        else:
            invalidInputError(args.framework == 'torch',
                              f"NBeats does not support tensorflow backend now.")
