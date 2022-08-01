# PPML Python Module Dev Guide
Before development, install all the required dependencies by `pip install -r requirements.txt`

## Environment Variables
In terminal, run `source prepare_env.sh`, you can see some environment variables printed. These are the environment variables used in developments.

In different IDE, you may need to copy these variables to some config file.
* PyCharm: In `Run Configuration`, copy them to `Environment Variables`.
* VSCode: copy them to `.env` or `.vscode/setting.json`

## Unit Test
In terminal run
```bash
bash test/run-pytests
```
If all the tests pass, your development environment is ready.

## Protobuf
The protobuf source is in `scala/ppml/src/main/proto`. If you want to add or modify protobuf functions, first modify protobuf source file.

Then run `bash generate-protobuf.sh`, the generated Python protobuf files will be in `py_proto`, copy them to some directory in `src` and add the directory to `PYTHONPATH` in `prepare_env.sh`.

You can consider `src/bigdl/ppml/fl/nn/generated` as an example.