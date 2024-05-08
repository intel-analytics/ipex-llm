# init ollama first
mkdir -p ~/ollama
cd ~/ollama
init-ollama
export OLLAMA_NUM_GPU=999
export ZES_ENABLE_SYSMAN=1

# start ollama service
(./ollama serve > ollama.log) &
