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


import subprocess
import os
import sys
import re
import json
import psycopg2
from datetime import datetime


class Workload:
    def __init__(self, name: str, script_path: str):
        self.name = name
        self.script_path = script_path

    def run(self):
        """Run a workload."""
        process = subprocess.run(f"bash {self.script_path}", stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, shell=True)
        if process.returncode != 0:
            err_msg = str(process.stderr, encoding='utf-8')
            print(err_msg)
            sys.exit(-1)
        else:
            output_msg = str(process.stdout, encoding='utf-8')
            print(output_msg)
            logs = self._extract_logs(output_msg)
            self._save_logs(logs)

    @staticmethod
    def _extract_logs(output: str):
        logs = [match.group(1) for match in re.finditer(">>>(.*?)<<<", output)]
        logs = [json.loads(log) for log in logs]
        return logs

    def _save_logs(self, logs: list):
        is_pr = True if os.environ.get('IS_PR') is not None else False
        timestamp = datetime.now()
        conn = psycopg2.connect(
            database=self._get_secret('DB_NAME'),
            user    =self._get_secret('DB_USER'),
            password=self._get_secret('DB_PASS'),
            host    =self._get_secret('DB_HOST'),
            port    =self._get_secret('DB_PORT')
        )

        cursor = conn.cursor()
        for log in logs:
            log['time'] = timestamp
            log['is_pr'] = is_pr
            sql = self._get_sql(log)
            print(sql)
            print(log)
            cursor.execute(sql, log)

        conn.commit()
        conn.close()

    def _get_secret(self, key: str):
        config_dir = os.environ.get('CONFIG_DIR')
        secret = open(os.path.join(config_dir, key), 'r').read()
        return secret

    def _get_sql(self, log: dict):
        keys = [key for key in log.keys()]
        fields = ", ".join(keys)
        place_holders = ", ".join([f"%({key})s" for key in keys])
        sql = f"INSERT INTO {self.name} ({fields}) VALUES ({place_holders})"
        return sql


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please pass the workload name as the first argument')
        sys.exit(-1)
    name = sys.argv[1]
    script_path = os.path.join("python/nano/benchmark", name, "run.sh")
    workload = Workload(name, script_path)
    workload.run()
