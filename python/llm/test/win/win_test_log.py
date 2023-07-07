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

import argparse
import os
from datetime import datetime

log_file_name = 'win_llm_test.log'

def create_new_log(logger_dir):
    archive_previous_log_file(logger_dir)
    new_log_path = os.path.join(logger_dir, log_file_name)
    with open(new_log_path, "w") as f:
        now = datetime.now()
        date_time = now.strftime("%Y%m%d-%H%M%S")
        f.write(date_time)
    return new_log_path


def archive_previous_log_file(logger_dir):
    log_file_path = os.path.join(logger_dir, log_file_name)
    if os.path.exists(log_file_path):
        with open(log_file_path) as f:
            time_info = f.readline().strip('\n')
        log_file_name_list = log_file_name.split('.')
        new_log_file_name = log_file_name_list[0] + "_" + time_info + "." + log_file_name_list[1]
        os.makedirs(os.path.join(logger_dir, "previous_logs"), exist_ok=True)
        os.rename(log_file_path, 
                os.path.join(logger_dir, "previous_logs", new_log_file_name))


def manage_logs(logger_dir):
    os.makedirs(logger_dir, exist_ok=True)
    new_log_path = create_new_log(logger_dir)
    return new_log_path


def main():
    parser = argparse.ArgumentParser(description='Win test logger')
    parser.add_argument('--logger_dir', type=str, 
                        default=r"C:\Users\obe\bigdl-llm-test\logs", required=True,
                        help='The directory to log files.')
    args = parser.parse_args()

    os.makedirs(args.logger_dir, exist_ok=True)
    create_new_log(args.logger_dir)


if __name__ == '__main__':
    main()
