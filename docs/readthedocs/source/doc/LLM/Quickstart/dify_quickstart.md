# Run Dify on Intel GPU

We recommend start the project following [Dify docs](https://docs.dify.ai/getting-started/install-self-hosted/local-source-code)
## Server Deployment
### Clone code
```bash
git clone https://github.com/langgenius/dify.git
```
### Installation of the basic environment
Server startup requires Python 3.10.x. Anaconda is recommended to create and manage python environment.  
```bash
conda create -n dify python=3.10
conda activate dify
cd api
cp .env.example .env
openssl rand -base64 42
sed -i 's/SECRET_KEY=.*/SECRET_KEY=<your_value>/' .env
pip install -r requirements.txt
```
### Prepare for redis, postgres, node and npm. 
* Install Redis by `sudo apt-get install redis-server`. Refer to [page](https://www.hostinger.com/tutorials/how-to-install-and-setup-redis-on-ubuntu/) to setup the Redis environment, including password, demon, etc. 
* install postgres by `sudo apt-get install postgres` and `sudo apt-get install postgres-client`. Setup username, create a database and grant previlidge according to [page](https://www.ruanyifeng.com/blog/2013/12/getting_started_with_postgresql.html)
* install npm and node by  `brew install node@20` according to [nodejs page](https://nodejs.org/en/download/package-manager)
> Note that set redis and postgres related environment in .env under dify/api/ and set web related environment variable in .env.local under dify/web

### Install Ollama
Please install ollama refer to [ollama quick start](./ollama_quickstart.md). Ensure that ollama could run successfully on Intel GPU. 

### Start service
1. Open the terminal and set `export no_proxy=localhost,127.0.0.1`
```bash
flask db upgrade
flask run --host 0.0.0.0 --port=5001 --debug
```
You will see log like below if successfully start the service. 
```
INFO:werkzeug:
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5001
* Running on http://10.239.44.83:5001
INFO:werkzeug:Press CTRL+C to quit
INFO:werkzeug: * Restarting with stat
WARNING:werkzeug: * Debugger is active!
INFO:werkzeug: * Debugger PIN: 227-697-894
```
2. Open another terminal and also set `export no_proxy=localhost,127.0.0.1`. 
If Linux system, use the command below. 
```bash
celery -A app.celery worker -P gevent -c 1 -Q dataset,generation,mail --loglevel INFO 
```
If windows system, use the command below. 
```bash
celery -A app.celery worker -P solo --without-gossip --without-mingle -Q dataset,generation,mail --loglevel INFO
```
3. Open another terminal and also set `export no_proxy=localhost,127.0.0.1`. Run the commands below to start the front-end service. 
```bash
cd web
npm install
npm run build
npm run start
```

## Example: RAG
See the demo of running dify with Ollama on an Intel Core Ultra laptop below.

<video src="https://llm-assets.readthedocs.io/en/latest/_images/dify-rag-small.mp4" width="100%" controls></video>



1. Set up the environment `export no_proxy=localhost,127.0.0.1` and start Ollama locally by `ollama serve`. 
2. Open http://localhost:3000 to view dify and change the model provider in setting including both LLM and embedding. For example, choose ollama. 
<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-p1.png" alt="image-p1" width=50%; />
</div>

3. Use text summarization workflow template from studio. 
<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-p2.png" alt="image-p2" width=50%; />
</div>

4. Add knowledge base and specify which type of embedding model to use. 
<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-p3.png" alt="image-p3" width=50%; />
</div>

5. Enter input and start to generate. You could find retrieval results and answers generated on the right. 
4. Add knowledge base and specify which type of embedding model to use. 
<div align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-p5.png" alt="image-20240221102252560" width=50%; />
</div>



