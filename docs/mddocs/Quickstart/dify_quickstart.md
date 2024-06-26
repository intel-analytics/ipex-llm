# Run Dify on Intel GPU


[**Dify**](https://dify.ai/) is an open-source production-ready LLM app development platform; by integrating it with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily leverage local LLMs running on Intel GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) for building complex AI workflows (e.g. RAG).  


*See the demo of a RAG workflow in Dify running LLaMA2-7B on Intel A770 GPU below.*

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/dify-rag-small.mp4"><img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-rag-small.png"/></a></td>
  </tr>
  <tr>
    <td align="center">You could also click <a href="https://llm-assets.readthedocs.io/en/latest/_images/dify-rag-small.mp4">here</a> to watch the demo video.</td>
  </tr>
</table>

## Table of Contents
- [Install and Start Ollama Service on Intel GPU](./dify_quickstart.md#1-install-and-start-ollama-service-on-intel-gpu)
- [Install and Start Dify](./dify_quickstart.md#2-install-and-start-dify)
- [How to Use Dify](./dify_quickstart.md#3-how-to-use-dify)



## Quickstart

### 1. Install and Start `Ollama` Service on Intel GPU 

Follow the steps in [Run Ollama on Intel GPU Guide](./ollama_quickstart.md) to install and run Ollama on Intel GPU. Ensure that `ollama serve` is running correctly and can be accessed through a local URL (e.g., `https://127.0.0.1:11434`) or a remote URL (e.g., `http://your_ip:11434`).

We recommend pulling the desired model before proceeding with Dify. For instance, to pull the LLaMA2-7B model, you can use the following command:

```bash
ollama pull llama2:7b
```

### 2. Install and Start `Dify`


#### 2.1 Download `Dify`

You can either clone the repository or download the source zip from [github](https://github.com/langgenius/dify/archive/refs/heads/main.zip):
```bash
git clone https://github.com/langgenius/dify.git
```

#### 2.2 Setup Redis and PostgreSQL

Next, deploy PostgreSQL and Redis. You can choose to utilize Docker, following the steps in the [Local Source Code Start Guide](https://docs.dify.ai/getting-started/install-self-hosted/local-source-code#clone-dify), or proceed without Docker using the following instructions:


- Install Redis by executing `sudo apt-get install redis-server`. Refer to [this guide](https://www.hostinger.com/tutorials/how-to-install-and-setup-redis-on-ubuntu/) for Redis environment setup, including password configuration and daemon settings.

- Install PostgreSQL by following either [the Official PostgreSQL Tutorial](https://www.postgresql.org/docs/current/tutorial.html) or [a PostgreSQL Quickstart Guide](https://www.digitalocean.com/community/tutorials/how-to-install-postgresql-on-ubuntu-20-04-quickstart). After installation, proceed with the following PostgreSQL commands for setting up Dify. These commands create a username/password for Dify (e.g., `dify_user`, change `'your_password'` as desired), create a new database named `dify`, and grant privileges:
    ```sql
    CREATE USER dify_user WITH PASSWORD 'your_password';
    CREATE DATABASE dify;
    GRANT ALL PRIVILEGES ON DATABASE dify TO dify_user;
    ```

Configure Redis and PostgreSQL settings in the `.env` file located under dify source folder `dify/api/`:

```bash dify/api/.env
### Example dify/api/.env
## Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_USERNAME=your_redis_user_name # change if needed
REDIS_PASSWORD=your_redis_password # change if needed
REDIS_DB=0

## postgreSQL settings
DB_USERNAME=dify_user # change if needed
DB_PASSWORD=your_dify_password # change if needed
DB_HOST=localhost
DB_PORT=5432
DB_DATABASE=dify # change if needed
```

#### 2.3 Server Deployment

Follow the steps in the [`Server Deployment` section in Local Source Code Start Guide](https://docs.dify.ai/getting-started/install-self-hosted/local-source-code#server-deployment) to deploy and start the Dify Server.

Upon successful deployment, you will see logs in the terminal similar to the following:


```bash
INFO:werkzeug:
* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:5001
* Running on http://10.239.44.83:5001
INFO:werkzeug:Press CTRL+C to quit
INFO:werkzeug: * Restarting with stat
WARNING:werkzeug: * Debugger is active!
INFO:werkzeug: * Debugger PIN: 227-697-894
```


#### 2.4 Deploy the frontend page

Refer to the instructions provided in the [`Deploy the frontend page` section in Local Source Code Start Guide](https://docs.dify.ai/getting-started/install-self-hosted/local-source-code#deploy-the-frontend-page) to deploy the frontend page.

Below is an example of environment variable configuration found in `dify/web/.env.local`:


```bash
# For production release, change this to PRODUCTION
NEXT_PUBLIC_DEPLOY_ENV=DEVELOPMENT
NEXT_PUBLIC_EDITION=SELF_HOSTED
NEXT_PUBLIC_API_PREFIX=http://localhost:5001/console/api
NEXT_PUBLIC_PUBLIC_API_PREFIX=http://localhost:5001/api
NEXT_PUBLIC_SENTRY_DSN=
```

> [!NOTE]
> If you encounter connection problems, you may run `export no_proxy=localhost,127.0.0.1` before starting API servcie, Worker service and frontend. 


### 3. How to Use `Dify`

For comprehensive usage instructions of Dify, please refer to the [Dify Documentation](https://docs.dify.ai/). In this section, we'll only highlight a few key steps for local LLM setup.


#### Setup Ollama

Open your browser and access the Dify UI at `http://localhost:3000`.


Configure the Ollama URL in `Settings > Model Providers > Ollama`. For detailed instructions on how to do this, see the [Ollama Guide in the Dify Documentation](https://docs.dify.ai/tutorials/model-configuration/ollama).


<p align="center"><a href="https://docs.dify.ai/~gitbook/image?url=https%3A%2F%2F3866086014-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FRncMhlfeYTrpujwzDIqw%252Fuploads%252Fgit-blob-351b275c8b6420ff85c77e67bf39a11aaf899b7b%252Follama-config-en.png%3Falt%3Dmedia&width=768&dpr=2&quality=100&sign=1ec95e72d9d0459384cce28665eb84ffd8ed59c906ab0fdb3f47fa67f61275dc"  target="_blank" align="center"><img src="https://docs.dify.ai/~gitbook/image?url=https%3A%2F%2F3866086014-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FRncMhlfeYTrpujwzDIqw%252Fuploads%252Fgit-blob-351b275c8b6420ff85c77e67bf39a11aaf899b7b%252Follama-config-en.png%3Falt%3Dmedia&width=768&dpr=2&quality=100&sign=1ec95e72d9d0459384cce28665eb84ffd8ed59c906ab0fdb3f47fa67f61275dc" alt="rag-menu" width="80%" align="center"></a></p>

Once Ollama is successfully connected, you will see a list of Ollama models similar to the following: 
<p align="center"><a href="https://llm-assets.readthedocs.io/en/latest/_images/dify-p1.png" target="_blank" align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-p1.png" alt="image-p1" width=100%; />
</a></p>



#### Run a simple RAG 

- Select the text summarization workflow template from the studio.
<p><a href="https://llm-assets.readthedocs.io/en/latest/_images/dify-p2.png" target="_blank" align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-p2.png" alt="image-p2" width=100%; align="center" />
</a></p>

- Add a knowledge base and specify the LLM or embedding model to use. 
<p><a href="https://llm-assets.readthedocs.io/en/latest/_images/dify-p3.png" target="_blank" align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-p3.png" alt="image-p3" width=100%; />
</a></p>

- Enter your input in the workflow and execute it. You'll find retrieval results and generated answers on the right.
<p align="center"><a href="https://llm-assets.readthedocs.io/en/latest/_images/dify-p5.png" target="_blank" align="center">
<img src="https://llm-assets.readthedocs.io/en/latest/_images/dify-p5.png" alt="image-20240221102252560" width=100%; align="center"/>
</a></p>


