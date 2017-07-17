## TensorTree

TensorTree class is used to decode a tensor to a tree structure.
The given input `content` is a tensor which encodes a constituency parse tree.
The tensor should have the following structure:

Each row of the tensor represents a tree node and the row number is node number.
For each row, except the last column, all other columns represent the children
node number of this node. Assume the value of a certain column of the row is not zero,
the value `p` means this node has a child whose node number is `p` (lies in the `p`-th)
row. Each leaf has a leaf number, in the tensor, the last column represents the leaf number.
Each leaf does not have any children, so all the columns of a leaf except the last should
be zero. If a node is the root, the last column should equal to `-1`.

Note: if any row for padding, the padding rows should be placed at the last rows with all
elements equal to `-1`.

eg. a tensor represents a binary tree:

```
[11, 10, -1;
 0, 0, 1;
 0, 0, 2;
 0, 0, 3;
 0, 0, 4;
 0, 0, 5;
 0, 0, 6;
 4, 5, 0;
 6, 7, 0;
 8, 9, 0;
 2, 3, 0;
 -1, -1, -1;
 -1, -1, -1]
```

**Parameters:**
* **content** the tensor to be encoded
