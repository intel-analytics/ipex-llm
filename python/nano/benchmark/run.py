import subprocess
import os
import sys
import re

class Workload:
    def __init__(self, name: str, script_path: str):
        self.name = name
        self.script_path = script_path

    def run(self):
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
        return logs

    def _save_logs(self, logs: list):
        for log in logs:
            print(log)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please pass the workload name as the first argument')
        sys.exit(-1)
    name = sys.argv[1]
    script_path = os.path.join("python/nano/benchmark", name, "run.sh")
    workload = Workload(name, script_path)
    workload.run()
