### Data
We use [Census]() data in this example

To simulate the scenario of two parties, we use select different features of Census data.

The original data has 15 columns. In preprocessing, some new feature are created from the combinations of some existed columns.

* data of client 1: `age`, `education`, `occupation`, cross columns: `edu_occ`, `age_edu_occ`
* data of client 2: `relationship`, `workclass`, `marital_status`
