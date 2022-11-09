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
import requests
import psycopg2
from datetime import datetime
from uuid import uuid1


class Workload:
    def __init__(self, name: str, script_path: str):
        self.name = name
        self.script_path = script_path

    def run(self):
        """Run a workload."""
        process = subprocess.run(f"bash {self.script_path}", stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, shell=True)
        uuid = '' # uuid=='' means running workload failed
        try:
            if process.returncode != 0:
                err_msg = str(process.stderr, encoding='utf-8')
                print(err_msg)
                raise Exception(f"Workload {self.name} failed to run !")
            else:
                output_msg = str(process.stdout, encoding='utf-8')
                print(f'Workload {self.name} finished successfully !')
                logs = self._extract_logs(output_msg)
                uuid = self._save_logs(logs)
        finally:
            # notify user if this benchmark is triggered by PR comments
            if os.environ.get('IS_COMMENTS') == 'true':
                self._notify(uuid)

    @staticmethod
    def _extract_logs(output: str):
        logs = [match.group(1) for match in re.finditer(">>>(.*?)<<<", output)]
        logs = [json.loads(log) for log in logs]
        return logs

    def _save_logs(self, logs: list):
        is_pr = True if os.environ.get('IS_PR') == 'true' else False
        timestamp = datetime.now()
        uuid = str(uuid1())
        conn = psycopg2.connect(
            database=self._get_secret('DB_NAME'),
            user    =self._get_secret('DB_USER'),
            password=self._get_secret('DB_PASS'),
            host    =self._get_secret('DB_HOST'),
            port    =self._get_secret('DB_PORT')
        )

        cursor = conn.cursor()
        for log in logs:
            other_fields = { 'time': timestamp, 'is_pr': is_pr, 'uuid': uuid }
            log = {**log, **other_fields}
            sql = self._get_sql(log)
            cursor.execute(sql, log)

        conn.commit()
        conn.close()
        return uuid

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

    def _notify(self, uuid: str):
        user = os.environ.get("USER")
        pr_url = os.environ.get("PR_URL")
        comment_url = os.environ.get("COMMENT_URL")
        job_url = os.environ.get("JOB_URL")
        token = os.environ.get("GITHUB_ACCESS_TOKEN")

        text = f"@{user} Your benchmark <{comment_url}> of workload `{self.name}` \
            has finished, see <{job_url}> for details\n\n"
        if uuid == '':
            text += "It seems there are some errors"
        else:
            text += f"Use `uuid='{uuid}'` to query the result"

        requests.post(
            url=pr_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"token {token}",
            },
            json={ "body": text }
        )

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python python/benchmark/run.py module workload')
        sys.exit(-1)
    module = sys.argv[1]
    name = sys.argv[2]
    script_path = os.path.join("python", module, "benchmark", name, "run.sh")
    workload = Workload(name, script_path)
    workload.run()