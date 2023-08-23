#!/bin/bash

# Attestation
if [ -z "$ATTESTATION" ]; then
  echo "[INFO] Attestation is disabled!"
  ATTESTATION="false"
fi

if [ "$ATTESTATION" ==  "true" ]; then
  if [ -e "/dev/tdx_guest" ]; then
    cd /opt
    bash /opt/attestation.sh
    bash /opt/temp_command_file
    if [ $? -ne 0 ]; then
      echo "[ERROR] Attestation Failed!"
      # exit 1
    fi
  else
      echo "TDX device not found!"
  fi
fi

mkdir -p /ppml/config && cd /ppml/config
cat > config.json <<EOF
{
    "model_specs_path": "/ppml/config/models.json",
    "models": {
        "Vicuna_7B_INT4": {
            "thread_num": 24,
            "model_is_encrypted": true,
            "avx512": true,
            "local": true
        }
    }
}
EOF

cat > models.json <<EOF
{
    "Vicuna_7B_INT4": {
        "family": "LLAMA:llamacpp",
        "path": {
            "AVX512": "/ppml/models/vicuna-7b-int4/bigdl_llm_vicuna_q4_0.bin.encrypted",
            "AVX2": "/ppml/models/vicuna-7b-int4/bigdl_llm_vicuna_q4_0.bin.encrypted"
        }
    }
}
EOF

mkdir -p /ppml/models/vicuna-7b-int4 && cd /ppml/models/vicuna-7b-int4 && wget https://tee-llm.oss-cn-beijing.aliyuncs.com/bigdl_llm_vicuna_q4_0.bin.encrypted

export SENTENCE_TRANSFORMERS_HOME="/ppml/models"
export PATH=$PATH

cd /ppml/ChatLLM
bash deploy.sh --config /ppml/config/config.json --host 0.0.0.0 --port 8083

# We do not have any arguments, just run bash
if [ "$#" == 0 ]; then
  echo "[INFO] no command is passed in"
  echo "[INFO] enter pass-through mode"
  # exec /usr/bin/tini -s -- "bash"
  tail -f /dev/null
else
  runtime_command="$@"
  exec $runtime_command
fi
