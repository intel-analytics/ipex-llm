# Trusted FL (Federated Learning)

SGX-based End-to-end Trusted FL platform

## ID & Feature align

Before we start Federated Learning, we need to align ID & Feature, and figure out portions of local data that will participate in later training stage.

Let RID1 and RID2 be randomized ID from party 1 and party 2.

## Vertical FL

Vertical FL training across multi-parties with different features.

Key features:

* FL Server in SGX
    * ID & feature align
    * Forward & backward aggregation
* Training node in SGX

## Horizontal FL

Horizontal FL training across multi-parties.

Key features:

* FL Server in SGX
   * ID & feature align (optional)
   * Weight/Gradient Aggregation in SGX
* Training Worker in SGX

## References

1. [Intel SGX](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html)
2. Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. 2019. Federated Machine Learning: Concept and Applications. ACM Trans. Intell. Syst. Technol. 10, 2, Article 12 (February 2019), 19 pages. DOI:https://doi.org/10.1145/3298981
