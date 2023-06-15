# +
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch import nn
import os
import json

# get distributed conf
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))

class PPMLContext:
    def __init__(self, type):
        self.type = type
        if type == "k8s":
            print("Using distributed PyTorch with {} backend".format(
                "GLOO"), flush=True)
            dist.init_process_group(backend=dist.Backend.GLOO)

    def set_training(self, model, train_dataloader, valid_dataloader = None):
        #get distributed model
        if self.type == "local":
            return model, train_dataloader, valid_dataloader
        elif self.type == "k8s":
            print("test----------")
            Distributor = nn.parallel.DistributedDataParallel
            model1 = Distributor(model, find_unused_parameters=True)

            #get distributed data_loader
            train_dataloader_kwargs = train_dataloader.__dict__.copy()  # 获取train_dataloader1的当前配置
            train_sampler = DistributedSampler(
                train_dataloader_kwargs['dataset'], num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False)
            train_dataloader1 = DataLoader(
                train_dataloader_kwargs['dataset'],
                batch_size=train_dataloader_kwargs['batch_size'],
                collate_fn=train_dataloader_kwargs['collate_fn'],
                sampler=train_sampler,
            )

            if valid_dataloader != None:
                valid_dataloader_kwargs = valid_dataloader.__dict__.copy()  # 获取train_dataloader1的当前配置
                valid_sampler = DistributedSampler(
                    valid_dataloader_kwargs['dataset'], num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, drop_last=False)
                valid_dataloader1 = DataLoader(
                    valid_dataloader_kwargs['dataset'],
                    batch_size=valid_dataloader_kwargs['batch_size'],
                    collate_fn=valid_dataloader_kwargs['collate_fn'],
                    sampler=valid_sampler,
                )
            return model1, train_dataloader1, valid_dataloader1
        else:
            raise ValueError("running type error, just support local or k8s")


# -
class PPMLConf:
    def __init__(self, k8s_enabled = True, sgx_enabled = True):
        self.conf = {}
        self.k8s_conf = {}
        self.k8s_env = {}
        self.volume_host = {}
        self.volume_nfs = {}
        self.volume_mount = {}
        self.main_script = {}
        self.k8s_enabled = k8s_enabled
        self.sgx_enabled = sgx_enabled
        self.init_k8s_conf()
        print("init")

    def exit(self):
        import sys
        sys.exit("success")

    def set(self, key, value, value1=None):
        if key.startswith("bigdl"):
            keys = key.split(".")
            # bigdl.ppml.k8s.env | conf | volume_mount |  volume_host | volume_nfs
            if keys[0] == "bigdl" and keys[1] == "ppml" and keys[2] == "k8s":
                if keys[3] == "env":
                    self.set_k8s_env(keys[4], value)
                elif keys[3] == "conf":
                    self.set_k8s_conf(keys[4], value)
                elif keys[3] == "volume_mount":
                    self.set_volume_mount(keys[4], value)
                elif keys[3] == "volume_host":
                    self.set_volume_host(keys[4], value)
                elif keys[3] == "volume_nfs":
                    self.set_volume_nfs(keys[4], value, value1)
            else:
                raise ValueError("ppml conf type error, just try again")
        else:
            self.conf[key] = value
        return self

    def set_k8s_conf(self, key, value):
        self.k8s_conf[key] = value
        return self

    def set_k8s_env(self, key, value):
        self.k8s_env[key] = value
        return self

    def set_volume_host(self, key, value):
        self.volume_host[key] = value
        return self

    def set_volume_nfs(self, key, ip, value):
        dict = {}
        dict["nfs_server"] = ip
        dict["nfs_path"] = value
        self.volume_nfs[key] = dict
        return self

    def set_volume_mount(self, key, value):
        self.volume_mount[key] = value
        return self

    def run_k8s(self):
        import k8s_deployment
        k8s_args = self.conf_to_args()
        k8s_deployment.run_k8s(k8s_args)

    def conf_to_args(self):
        args = []
        if self.k8s_enabled == True:
            main_script = self.conf["main_script"]
            if (main_script == ""):
                raise ValueError("main_script is empty, just try again")
            current_dir = os.getcwd()
            self.set("main_script", current_dir + "/" + main_script + ".ipynb")
            self.set("main_script_args", "--test 2")
            for key, value in self.k8s_conf.items():
                args.append("--" + key)
                args.append(value)
            for key, value in self.k8s_env.items():
                args.append("--env")
                args.append(key)
                args.append(value)
            for key, value in self.volume_host.items():
                args.append("--volume")
                data = {
                    "name": key,
                    "hostPath": {
                        "path": value
                    }
                }
                json_string = json.dumps(data)
                args.append(json_string)
            for key, value in self.volume_nfs.items():
                args.append("--volume")
                data = {
                    "name": key,
                    "nfs": {
                        "server": value["nfs_server"],
                        "path": value["nfs_path"]
                    }
                }
                json_string = json.dumps(data)
                args.append(json_string)
            for key, value in self.volume_mount.items():
                args.append("--volume_mount")
                data = {
                    "mountPath": value,
                    "name": key
                }
                json_string = json.dumps(data)
                args.append(json_string)
        for key, value in self.conf.items():
            args.append("--" + key)
            args.append(value)
        print(args)
        return args




    def init_k8s_conf(self):
        if self.k8s_enabled:
            # set device-plugin and aesm
            self.set("bigdl.ppml.k8s.volume_host.device-plugin", "/var/lib/kubelet/device-plugins") \
                .set("bigdl.ppml.k8s.volume_mount.device-plugin", "/var/lib/kubelet/device-plugins") \
                .set("bigdl.ppml.k8s.volume_host.aesm-socket", "/var/run/aesmd/aesm.socket") \
                .set("bigdl.ppml.k8s.volume_mount.aesm-socket", "/var/run/aesmd/aesm.socket")

            # set conf
            self.set("bigdl.ppml.k8s.env.http_proxy", "http://child-prc.intel.com:913/") \
                .set("bigdl.ppml.k8s.env.https_proxy", "http://child-prc.intel.com:913/") \
                .set("bigdl.ppml.k8s.env.no_proxy", "10.239.45.10:8081,10.112.231.51,10.239.45.10,172.168.0.*") \
                .set("bigdl.ppml.k8s.conf.namespace", "default") \
                .set("bigdl.ppml.k8s.conf.image", "intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-ref:2.4.0-SNAPSHOT") \
                .set("bigdl.ppml.k8s.conf.driver_port", "29500")


        if self.sgx_enabled:
            self.set("bigdl.ppml.k8s.env.SGX_ENABLED", "true")
        else:
            self.set("bigdl.ppml.k8s.env.SGX_ENABLED", "false")
