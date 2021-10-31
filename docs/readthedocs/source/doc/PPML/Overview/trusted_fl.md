# Trusted FL (Federated Learning)

SGX-based E2e Trusted FL platform

## ID & Feature align

Before we start Federated Learning, we need to align ID & Feature, and figure out portions of local data that will participate in later training stage.

Let RID1 and RID2 be randomized ID from party 1 and party 2.

<div align="center">
   <p align="center"> <img src="../docs/image/id_align.png" height=180px; weight=650px;"><br></p>
</div>


## Vertical FL

Key features:

* FL Server in SGX
    * id & feature align
    * Forward & backward
* Training node in SGX

<div align="center">
   <p align="center"> <img src="../docs/image/vfl.png" height=360px; weight=800px;"><br></p>
</div>

## Horizontal FL

Key features:

* id & feature align (optional)
* Parameter Server in SGX
* Training Worker in SGX

<div align="center">
   <p align="center"> <img src="../docs/image/hfl.png" height=360px; weight=800px;"><br></p>
</div>

## References

1. [Intel SGX](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html)
2. Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. 2019. Federated Machine Learning: Concept and Applications. ACM Trans. Intell. Syst. Technol. 10, 2, Article 12 (February 2019), 19 pages. DOI:https://doi.org/10.1145/3298981
