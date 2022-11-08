# PPML Installation

---

#### OS requirement


```eval_rst
.. note::
    **Required Hardware**:

     PPML's key features (except CKKS) are mainly built on Intel SGX. Intel SGX is a hardware feature provided by Intel's CPU. [Check if your CPU has SGX feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors.html).
```
```eval_rst
.. note::
    **Supported OS**:

     Chronos is thoroughly tested on Ubuntu (18.04/20.04), and should works fine on CentOS/Redhat 8.
```

```eval_rst
.. mermaid::
   
   graphe TD
      A(Node A)
      B([Node B])

      A -- points to --> B
      A --> C{{Node C}}

      classDef blue color:#0171c3;
      class B,C blue;
```


#### Install SGX Driver

#### Install SGX Service (Optional)

Install aesm

Install PCCS

#### Install Kubernetes SGX Plugin

#### Install KMS
