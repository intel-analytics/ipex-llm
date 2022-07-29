conda create -y -n chronos python=3.7 setuptools=58.0.4 && \
source activate chronos 
#MODE choice:
#pytorch-only,pytorch-onnx,pytorch-automl,pytorch-dist,pytorch-automl-dist
#tf-only,tf-automl,tf-dist,tf-automl-dist
#default,automl-prophet,all
case $1 in
	default)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	all)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[all] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	pytorch-only)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[pytorch] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	pytorch-onnx)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[pytorch] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0 && \
		pip install --no-cache-dir onnxruntime==1.6.0 && \
		pip install --no-cache-dir onnx && \
		pip install --no-cache-dir prompt 
		;;
	pytorch-automl)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[pytorch,automl] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	pytorch-dist)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[pytorch,distributed] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	pytorch-automl-dist)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[pytorch,distributed,automl] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	tf-only)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[tensorflow] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	tf-automl)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[tensorflow,automl] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	tf-dist)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[tensorflow,distributed] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	tf-automl-dist)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[tensorflow,distributed,automl] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0
		;;
	automl-prophet)
		pip install --no-cache-dir numpy==1.19.5 && \
		pip install --no-cache-dir ray==1.9.2 ray[tune]==1.9.2 ray[default]==1.9.2 && \
		pip install --no-cache-dir --pre bigdl-chronos[automl] && \
		pip install --no-cache-dir scipy==1.5.4 && \
		pip install --no-cache-dir prometheus_pandas==0.3.1 && \
		pip install --no-cache-dir jupyter==1.0.0 &&\
		pip install --no-cache-dir prophet==1.1.0 &&\
		pip install --no-cache-dir pmdarima==1.8.4 
		;;
esac

