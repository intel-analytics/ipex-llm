/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2020 Invisible Things Lab
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

#include "avl_tree.h"

#include <stddef.h>

#include "api.h"
#include "assert.h"

static void avl_tree_init_node(struct avl_tree_node* node) {
    node->left    = NULL;
    node->right   = NULL;
    node->parent  = NULL;
    node->balance = 0;
}

/* Inserts a node into tree, but leaves it unbalanced, i.e. all nodes on path from root to newly
 * inserted node could have their balance field off by +1/-1 */
static void avl_tree_insert_unbalanced(struct avl_tree* tree,
                                       struct avl_tree_node* node_to_insert) {
    assert(tree);
    assert(tree->root);
    assert(node_to_insert);

    struct avl_tree_node* node = tree->root;

    while (1) {
        if (tree->cmp(node_to_insert, node)) {
            if (!node->left) {
                node->left = node_to_insert;
                node_to_insert->parent = node;
                return;
            } else {
                node = node->left;
            }
        } else {
            if (!node->right) {
                node->right = node_to_insert;
                node_to_insert->parent = node;
                return;
            } else {
                node = node->right;
            }
        }
    }
}

/* Replaces `old_node` with `new_node` fixing all necessary links. `parent` must be the parent of
 * `old_node` before call to this. */
static void fixup_link(struct avl_tree_node* old_node, struct avl_tree_node* new_node,
                       struct avl_tree_node* parent) {
    if (parent) {
        if (parent->left == old_node) {
            parent->left = new_node;
        } else {
            assert(parent->right == old_node);
            parent->right = new_node;
        }
    }

    if (new_node) {
        new_node->parent = parent;
    }
}

/*
 * The next 4 functions do rotations (rot1 - single, rot2 - double, which is a concatenation of two
 * single rotations). L stands for left (counterclockwise) rotation and R for right (clockwise).
 * The naming convention is: `p` is topmost node and parent of `q`, which in turn is parent of `r`.
 */

static void rot1L(struct avl_tree_node* q, struct avl_tree_node* p) {
    assert(q->parent == p);
    assert(p->right == q);
    assert(q->balance == 1 || q->balance == 0);
    assert(p->balance == 2);

    fixup_link(/*old_node=*/p, /*new_node=*/q, /*parent=*/p->parent);

    p->right = q->left;
    if (q->left) {
        q->left->parent = p;
    }

    q->left = p;
    p->parent = q;

    if (q->balance == 1) {
        p->balance = 0;
        q->balance = 0;
    } else { // q->balance == 0
        p->balance = 1;
        q->balance = -1;
    }
}

static void rot1R(struct avl_tree_node* q, struct avl_tree_node* p) {
    assert(q->parent == p);
    assert(p->left == q);
    assert(q->balance == -1 || q->balance == 0);
    assert(p->balance == -2);

    fixup_link(/*old_node=*/p, /*new_node=*/q, /*parent=*/p->parent);

    p->left = q->right;
    if (q->right) {
        q->right->parent = p;
    }

    q->right = p;
    p->parent = q;

    if (q->balance == -1) {
        p->balance = 0;
        q->balance = 0;
    } else { // q->balance == 0
        p->balance = -1;
        q->balance = 1;
    }
}

static void rot2RL(struct avl_tree_node* r, struct avl_tree_node* q, struct avl_tree_node* p) {
    assert(q->parent == p);
    assert(p->right == q);
    assert(q->balance == -1);
    assert(p->balance == 2);

    assert(r->parent == q);
    assert(q->left == r);
    assert(-1 <= r->balance && r->balance <= 1);

    fixup_link(/*old_node=*/p, /*new_node=*/r, /*parent=*/p->parent);

    p->right = r->left;
    if (r->left) {
        r->left->parent = p;
    }

    q->left = r->right;
    if (r->right) {
        r->right->parent = q;
    }

    r->left = p;
    p->parent = r;

    r->right = q;
    q->parent = r;

    if (r->balance == -1) {
        p->balance = 0;
        q->balance = 1;
    } else if (r->balance == 0) {
        p->balance = 0;
        q->balance = 0;
    } else { // r->balance == 1
        p->balance = -1;
        q->balance = 0;
    }
    r->balance = 0;
}

static void rot2LR(struct avl_tree_node* r, struct avl_tree_node* q, struct avl_tree_node* p) {
    assert(q->parent == p);
    assert(p->left == q);
    assert(q->balance == 1);
    assert(p->balance == -2);

    assert(r->parent == q);
    assert(q->right == r);
    assert(-1 <= r->balance && r->balance <= 1);

    fixup_link(/*old_node=*/p, /*new_node=*/r, /*parent=*/p->parent);

    q->right = r->left;
    if (r->left) {
        r->left->parent = q;
    }

    p->left = r->right;
    if (r->right) {
        r->right->parent = p;
    }

    r->left = q;
    q->parent = r;

    r->right = p;
    p->parent = r;

    if (r->balance == -1) {
        q->balance = 0;
        p->balance = 1;
    } else if (r->balance == 0) {
        q->balance = 0;
        p->balance = 0;
    } else { // r->balance == 1
        q->balance = -1;
        p->balance = 0;
    }
    r->balance = 0;
}

/* Does appropriate rotation of node, which mush have disturbed balance (i.e. +2/-2).
 * Returns whether height might have changed and sets `new_root_ptr` to root of this subtree after
 * rotation. */
static bool avl_tree_do_balance(struct avl_tree_node* node, struct avl_tree_node** new_root_ptr) {
    assert(node->balance == -2 || node->balance == 2);

    struct avl_tree_node* child = NULL;
    bool ret;

    if (node->balance < 0) { // node->balance == -2
        child = node->left;
        if (child->balance == 1) {
            assert(child->right);
            *new_root_ptr = child->right;
            rot2LR(child->right, child, node);
            return true;
        } else { // child->balance <= 0
            *new_root_ptr = child;
            ret = child->balance != 0;
            rot1R(child, node);
            return ret;
        }
    } else { // node->balance == 2
        child = node->right;
        if (child->balance >= 0) {
            *new_root_ptr = child;
            ret = child->balance != 0;
            rot1L(child, node);
            return ret;
        } else { // child->balance == -1
            assert(child->left);
            *new_root_ptr = child->left;
            rot2RL(child->left, child, node);
            return true;
        }
    }
}

enum side {
    LEFT,
    RIGHT
};

/*
 * Balances the tree going from `node` upwards, to the tree root, stopping if a subtree height
 * did not change.
 * `side` indicates which child of `node` had its height changed.
 * `height_increased` is false if the subtree height decreased, true if increased.
 *
 * Returns the root of the subtree that balancing stopped at.
 */
static struct avl_tree_node* avl_tree_balance(struct avl_tree_node* node, enum side side,
                                              bool height_increased) {
    assert(node);

    while (1) {
        bool height_changed = true;

        if (side == LEFT) {
            if (height_increased) {
                height_changed = node->balance <= 0;
                node->balance -= 1;
            } else {
                height_changed = node->balance < 0;
                node->balance += 1;
            }
        } else {
            assert(side == RIGHT);
            if (height_increased) {
                height_changed = node->balance >= 0;
                node->balance += 1;
            } else {
                height_changed = node->balance > 0;
                node->balance -= 1;
            }
        }

        assert(-2 <= node->balance && node->balance <= 2);
        if (node->balance == -2 || node->balance == 2) {
            height_changed = avl_tree_do_balance(node, &node);
            /* On inserting height never changes. */
            height_changed = height_increased ? false : height_changed;
        }

        /* This sub-tree is balanced, but its height might have changed. */
        if (!height_changed || !node->parent) {
            return node;
        }

        if (node->parent->left == node) {
            side = LEFT;
        } else {
            assert(node->parent->right == node);
            side = RIGHT;
        }
        node = node->parent;
    }
}

void avl_tree_insert(struct avl_tree* tree, struct avl_tree_node* node) {
    avl_tree_init_node(node);

    /* Inserting into an empty tree. */
    if (!tree->root) {
        tree->root = node;
        return;
    }

    avl_tree_insert_unbalanced(tree, node);

    assert(node->parent);

    struct avl_tree_node* new_root;

    if (node->parent->left == node) {
        new_root = avl_tree_balance(node->parent, LEFT, /*height_increased=*/true);
    } else {
        assert(node->parent->right == node);
        new_root = avl_tree_balance(node->parent, RIGHT, /*height_increased=*/true);
    }

    if (!new_root->parent) {
        tree->root = new_root;
    }
}

void avl_tree_swap_node(struct avl_tree* tree, struct avl_tree_node* old_node,
                        struct avl_tree_node* new_node) {
    assert(tree->cmp(old_node, new_node) && tree->cmp(new_node, old_node));

    avl_tree_init_node(new_node);

    fixup_link(/*old_node=*/old_node, /*new_node=*/new_node, /*parent=*/old_node->parent);

    new_node->left = old_node->left;
    if (new_node->left) {
        new_node->left->parent = new_node;
    }
    new_node->right = old_node->right;
    if (new_node->right) {
        new_node->right->parent = new_node;
    }

    new_node->balance = old_node->balance;

    if (tree->root == old_node) {
        tree->root = new_node;
    }
}

struct avl_tree_node* avl_tree_prev(struct avl_tree_node* node) {
    if (node->left) {
        node = node->left;
        while (node->right) {
            node = node->right;
        }
        return node;
    }
    while (node->parent && node->parent->left == node) {
        node = node->parent;
    }
    return node->parent;
}

struct avl_tree_node* avl_tree_next(struct avl_tree_node* node) {
    if (node->right) {
        node = node->right;
        while (node->left) {
            node = node->left;
        }
        return node;
    }
    while (node->parent && node->parent->right == node) {
        node = node->parent;
    }
    return node->parent;
}

struct avl_tree_node* avl_tree_first(struct avl_tree* tree) {
    struct avl_tree_node* node = tree->root;
    if (!node) {
        return NULL;
    }

    while (node->left) {
        node = node->left;
    }

    return node;
}

struct avl_tree_node* avl_tree_last(struct avl_tree* tree) {
    struct avl_tree_node* node = tree->root;
    if (!node) {
        return NULL;
    }

    while (node->right) {
        node = node->right;
    }

    return node;
}

void avl_tree_delete(struct avl_tree* tree, struct avl_tree_node* node) {
    /* If `node` has both children, swap it with the next node. This might temporarily disturb
     * the tree order, but only between `node` and `next`, which is ok, since we are about to
     * remove `node` from the tree completely.
     * This is done so that `node` has 1 child at most (if a node has 2 children, then the next
     * node cannot have its left child). */
    if (node->left && node->right) {
        struct avl_tree_node* next = avl_tree_next(node);
        assert(next->balance == 0 || next->balance == 1);
        if (next->right) {
            assert(next->right->balance == 0);
            assert(!next->right->left);
            assert(!next->right->right);
        }
        assert(next->parent);

        struct avl_tree_node* tmp_right = next->right;
        struct avl_tree_node* tmp_parent = next->parent;
        signed char tmp_balance = next->balance;

        fixup_link(/*old_node=*/node, /*new_node=*/next, /*parent=*/node->parent);
        /* In this order it works even if both next->left and next->right are NULL pointers,
         * because node->left is not NULL here. */
        fixup_link(/*old_node=*/next->left, /*new_node=*/node->left, /*parent=*/next);
        if (next == node->right) {
            next->right = node;
            node->parent = next;
        } else {
            fixup_link(/*old_node=*/next->right, /*new_node=*/node->right, /*parent=*/next);
            fixup_link(/*old_node=*/next, /*new_node=*/node, /*parent=*/tmp_parent);
        }
        node->left = NULL;
        fixup_link(/*old_node=*/node->right, /*new_node=*/tmp_right, /*parent=*/node);

        next->balance = node->balance;
        node->balance = tmp_balance;

        if (tree->root == node) {
            tree->root = next;
        }
    }

    assert(!(node->left && node->right));

    /* This initialization value has no meaning, it's just here to keep GCC happy. */
    enum side side = LEFT;

    if (node->parent) {
        if (node->parent->left == node) {
            side = LEFT;
        } else {
            assert(node->parent->right == node);
            side = RIGHT;
        }
    }

    struct avl_tree_node* new_root = NULL;

    /* Remove `node` from the tree. */
    if (!node->left && !node->right) {
        new_root = NULL;
        fixup_link(/*old_node=*/node, /*new_node=*/NULL, /*parent=*/node->parent);
    } else if (node->left && !node->right) {
        new_root = node->left;
        fixup_link(/*old_node=*/node, /*new_node=*/node->left, /*parent=*/node->parent);
    } else {
        assert(!node->left && node->right);
        new_root = node->right;
        fixup_link(/*old_node=*/node, /*new_node=*/node->right, /*parent=*/node->parent);
    }

    /* After removal the tree might need balancing. */
    if (node->parent) {
        new_root = avl_tree_balance(node->parent, side, /*height_increased=*/false);
    }

    if ((new_root && !new_root->parent) || !node->parent) {
        tree->root = new_root;
    }
}

static struct avl_tree_node* avl_tree_find_fn_to(struct avl_tree* tree,
                                                 struct avl_tree_node* cmp_arg,
                                                 bool cmp(struct avl_tree_node*,
                                                          struct avl_tree_node*)) {
    struct avl_tree_node* node = tree->root;

    while (node) {
        bool x = cmp(cmp_arg, node);
        if (x) {
            if (cmp(node, cmp_arg)) {
                return node;
            }
            node = node->left;
        } else {
            node = node->right;
        }
    }

    return NULL;
}

struct avl_tree_node* avl_tree_find(struct avl_tree* tree, struct avl_tree_node* node) {
    return avl_tree_find_fn_to(tree, node, tree->cmp);
}

struct avl_tree_node* avl_tree_lower_bound_fn(struct avl_tree* tree, void* cmp_arg,
                                              bool cmp(void*, struct avl_tree_node*)) {
    struct avl_tree_node* node = tree->root;
    struct avl_tree_node* ret = NULL;

    while (node) {
        if (cmp(cmp_arg, node)) {
            ret = node;
            node = node->left;
        } else {
            node = node->right;
        }
    }

    return ret;
}

struct avl_tree_node* avl_tree_lower_bound(struct avl_tree* tree, struct avl_tree_node* cmp_arg) {
    static_assert(SAME_TYPE(tree->cmp, bool (*)(struct avl_tree_node*, struct avl_tree_node*)),
                  "If you change this function type, make sure the code below works properly!");
    return avl_tree_lower_bound_fn(tree, cmp_arg,
                                   (bool (*)(void*, struct avl_tree_node*))tree->cmp);
}

/* This function returns whether a tree with root in `node` is avl-balanced and updates `*size`
 * with height of the tree. */
static bool avl_tree_is_balanced_size(struct avl_tree_node* node, size_t* size) {
    if (!node) {
        *size = 0;
        return true;
    }

    size_t a = 0;
    size_t b = 0;

    bool ret = avl_tree_is_balanced_size(node->left, &a);
    ret &= avl_tree_is_balanced_size(node->right, &b);

    if (a < b) {
        ret &= (b - a) == 1;
        ret &= node->balance == 1;
        *size = b;
    } else if (a == b) {
        ret &= node->balance == 0;
        *size = a;
    } else { // a > b
        ret &= (a - b) == 1;
        ret &= node->balance == -1;
        *size = a;
    }

    *size += 1;
    return ret;
}

bool debug_avl_tree_is_balanced(struct avl_tree* tree) {
    size_t s;
    return avl_tree_is_balanced_size(tree->root, &s);
}
