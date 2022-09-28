### Data
We use [Census]() data in this example

To simulate the scenario of two parties, we use a [script]() to divide the training data into 2 parts with different columns.

The original data has 15 columns. In preprocessing, some new columns are created from the combinations of some existed columns. Considering this process, we divide the data into

* data of client 1: `age`, `workclass`, `fnlwgt`, `education`, `education_num`, `marital_status`, `occupation`
* data of client 2: `relationship`, `race`, `gender`, `capital_gain`, `capital_loss`, `hours_per_week`, `native_country`, `income_bracket`
