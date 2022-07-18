#ifndef AVL_TREE_H
#define AVL_TREE_H

#include <stdbool.h>

/*
 * Example usage of this module:
 *
 * struct S {
 *     int key;
 *     struct avl_tree_node node;
 * };
 *
 * bool cmp1(struct avl_tree_node* a, struct avl_tree_node* b) {
 *     struct S* _a = container_of(a, struct S, node);
 *     struct S* _b = container_of(b, struct S, node);
 *     return _a->key <= _b->key;
 * }
 *
 * int cmp2(void* v, struct avl_tree_node* n) {
 *     int _v = *(int*)v;
 *     int _n = container_of(n, struct S, node)->key;
 *     if (_v < _n) {
 *         return -1;
 *     } else if (_v == _n) {
 *         return 0;
 *     } else { // _v > _n
 *         return 1;
 *     }
 * }
 *
 * struct avl_tree tree = { .root = NULL, .cmp = cmp };
 *
 * struct S element = { .key = 42 };
 * avl_tree_insert(&tree, &element.node);
 *
 * int v = 13;
 * struct avl_tree_node* found = avl_tree_lower_bound(&tree, &v, cmp2);
 */

struct avl_tree_node {
    struct avl_tree_node* left;
    struct avl_tree_node* right;
    struct avl_tree_node* parent;
    signed char balance; // tree_height(right) - tree_height(left)
};

struct avl_tree {
    /* `root` points to the root node of the tree or is NULL for an empty tree. */
    struct avl_tree_node* root;
    /* This should be a total order (<=) on tree nodes. If two elements compare equal, the newer
     * will be on the left (side of smaller elements) from the older one. */
    bool (*cmp)(struct avl_tree_node*, struct avl_tree_node*);
};

void avl_tree_insert(struct avl_tree* tree, struct avl_tree_node* node);
void avl_tree_delete(struct avl_tree* tree, struct avl_tree_node* node);

/*
 * This function swaps `new_node` in place of `old_node`. `new_node` must not be in any tree (i.e.
 * it should really be a new node) and they both should compare equal with respect to `tree.cmp` or
 * bad things will happen. You have been warned. Probably the only usecase of this function is to
 * optimize delete + insert of a node with the same key.
 */
void avl_tree_swap_node(struct avl_tree* tree, struct avl_tree_node* old_node,
                        struct avl_tree_node* new_node);

/* These functions return respectively previous and next node or NULL if such does not exist.
 * O(log(n)) in worst case, but amortized O(1). */
struct avl_tree_node* avl_tree_prev(struct avl_tree_node* node);
struct avl_tree_node* avl_tree_next(struct avl_tree_node* node);

struct avl_tree_node* avl_tree_first(struct avl_tree* tree);
struct avl_tree_node* avl_tree_last(struct avl_tree* tree);

/*
 * Find a node that compares equal to `cmp_arg`. If `tree` has multiple nodes that compare equal,
 * you could get *any* of them.
 * `cmp_arg` does not need to (and usually will not) be in `tree`, it is only passed as
 * an argument to comparison function (`tree->cmp`).
 */
struct avl_tree_node* avl_tree_find(struct avl_tree* tree, struct avl_tree_node* cmp_arg);

/*
 * Returns the smallest element in `tree` that is greater or equal to `cmp_arg`, i.e. for which
 * `cmp(cmp_arg, node)` is true. Note that if multiple elements compare equal to `cmp_arg`
 * the lately inserted will be returned (the one furthest on the left in the tree).
 * cmp(cmp_arg, node) should return:
 * - true if cmp_arg <= node
 * - false otherwise (if cmp_arg > node)
 * It must also be compatible with tree->cmp i.e. cmp(f(a), b) == tree->cmp(a, b) for all a, b,
 * where f is some function changing a tree node to whatever the type of first argument to `cmp` is.
 */
struct avl_tree_node* avl_tree_lower_bound_fn(struct avl_tree* tree, void* cmp_arg,
                                              bool cmp(void*, struct avl_tree_node*));
struct avl_tree_node* avl_tree_lower_bound(struct avl_tree* tree, struct avl_tree_node* cmp_arg);

bool debug_avl_tree_is_balanced(struct avl_tree* tree);

#endif // AVL_TREE_H
