#include "avl_tree.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdnoreturn.h>

#include "api.h"
#include "assert.h"
#include "pal.h"
#include "pal_regression.h"

#define EXIT_UNBALANCED()                                 \
    do {                                                  \
        pal_printf("Unbalanced tree at: %u\n", __LINE__); \
        DkProcessExit(1);                                 \
    } while (0)

static uint32_t _seed;

static void srand(uint32_t seed) {
    _seed = seed;
}

/* source: https://elixir.bootlin.com/glibc/glibc-2.31/source/stdlib/rand_r.c */
static int32_t rand(void) {
    int32_t result;

    _seed *= 1103515245;
    _seed += 12345;
    result = (uint32_t)(_seed / 65536) % 2048;

    _seed *= 1103515245;
    _seed += 12345;
    result <<= 10;
    result ^= (uint32_t)(_seed / 65536) % 1024;

    _seed *= 1103515245;
    _seed += 12345;
    result <<= 10;
    result ^= (uint32_t)(_seed / 65536) % 1024;

    return result;
}

struct A {
    struct avl_tree_node node;
    int64_t key;
    bool freed;
};

static struct A* node2struct(struct avl_tree_node* node) {
    return container_of(node, struct A, node);
}

static bool cmp(struct avl_tree_node* x, struct avl_tree_node* y) {
    return node2struct(x)->key <= node2struct(y)->key;
}

static bool cmp_gen(void* x, struct avl_tree_node* y) {
    return *(int64_t*)x <= node2struct(y)->key;
}

#define ELEMENTS_COUNT 0x1000
#define RAND_DEL_COUNT 0x100
static struct avl_tree tree = {.root = NULL, .cmp = cmp};
static struct A t[ELEMENTS_COUNT];

__attribute__((unused)) static void debug_print(struct avl_tree_node* node) {
    if (!node) {
        pal_printf("LEAF");
        return;
    }
    pal_printf("%ld (", node2struct(node)->key);
    debug_print(node->left);
    pal_printf(") (");
    debug_print(node->right);
    pal_printf(")");
}

static size_t get_tree_size(struct avl_tree_node* node) {
    if (!node) {
        return 0;
    }

    return get_tree_size(node->left) + 1 + get_tree_size(node->right);
}

static void try_node_swap(struct avl_tree_node* node, struct avl_tree_node* swap_node) {
    avl_tree_swap_node(&tree, node, swap_node);
    node->left   = (void*)1;
    node->right  = (void*)2;
    node->parent = (void*)3;
    if (!debug_avl_tree_is_balanced(&tree)) {
        EXIT_UNBALANCED();
    }
    size_t size = get_tree_size(tree.root);
    if (size != ELEMENTS_COUNT) {
        pal_printf("Tree has %lu elements instead of %u!", size, ELEMENTS_COUNT);
        DkProcessExit(1);
    }

    avl_tree_swap_node(&tree, swap_node, node);
    swap_node->left   = (void*)1;
    swap_node->right  = (void*)2;
    swap_node->parent = (void*)3;
    if (!debug_avl_tree_is_balanced(&tree)) {
        EXIT_UNBALANCED();
    }
    size = get_tree_size(tree.root);
    if (size != ELEMENTS_COUNT) {
        pal_printf("Tree has %lu elements instead of %u!", size, ELEMENTS_COUNT);
        DkProcessExit(1);
    }
}

static void do_test(int32_t (*get_num)(void)) {
    size_t i;

    for (i = 0; i < ELEMENTS_COUNT; i++) {
        t[i].key   = get_num();
        t[i].freed = false;
        avl_tree_insert(&tree, &t[i].node);
        if (!debug_avl_tree_is_balanced(&tree)) {
            EXIT_UNBALANCED();
        }
    }

    size_t size = get_tree_size(tree.root);
    if (size != ELEMENTS_COUNT) {
        pal_printf("Tree has %lu elements instead of %u!", size, ELEMENTS_COUNT);
        DkProcessExit(1);
    }

    struct avl_tree_node* node = avl_tree_first(&tree);
    struct avl_tree_node* prev = NULL;
    size = 0;
    while (node) {
        if (prev && node2struct(prev)->key > node2struct(node)->key) {
            pal_printf("Found two elements in wrong order!\n");
            DkProcessExit(1);
        }
        prev = node;
        node = avl_tree_next(node);
        size++;
    }
    if (size != ELEMENTS_COUNT) {
        pal_printf("Tree iteration from the beginning walked through %lu elements instead of %u!",
                   size, ELEMENTS_COUNT);
        DkProcessExit(1);
    }

    node = avl_tree_last(&tree);
    struct avl_tree_node* next = NULL;
    size = 0;
    while (node) {
        if (next && node2struct(node)->key > node2struct(next)->key) {
            pal_printf("Found two elements in wrong order while iterating backwards!\n");
            DkProcessExit(1);
        }
        next = node;
        node = avl_tree_prev(node);
        size++;
    }
    if (size != ELEMENTS_COUNT) {
        pal_printf("Tree iteration from the end walked through %lu elements instead of %u!", size,
                   ELEMENTS_COUNT);
        DkProcessExit(1);
    }

    static_assert(ELEMENTS_COUNT >= 3, "This code needs at least 3 elements in the tree!");

    node = tree.root->left;
    struct A swap_node = {.key = node2struct(node)->key};
    try_node_swap(node, &swap_node.node);

    node = tree.root;
    swap_node.key = node2struct(node)->key;
    try_node_swap(node, &swap_node.node);

    node = tree.root->left;
    while (node->right) {
        node = node->right;
    }

    int64_t val = node2struct(node)->key;
    struct avl_tree_node* found_node = avl_tree_lower_bound(&tree, node);
    if (!found_node || node2struct(found_node)->key != val) {
        pal_printf("avl_tree_lower_bound has not found existing node %ld, but returned ", val);
        if (found_node) {
            pal_printf("%ld", node2struct(found_node)->key);
        } else {
            pal_printf("NULL");
        }
        pal_printf("\n");
        DkProcessExit(1);
    }

    /* get_num returns int32_t, but tmp.key is a int64_t, so this cannot overflow. */
    struct A tmp = {.key = val + 100};
    avl_tree_insert(&tree, &tmp.node);
    if (!debug_avl_tree_is_balanced(&tree)) {
        EXIT_UNBALANCED();
    }

    val += 1;
    found_node = avl_tree_lower_bound_fn(&tree, &val, cmp_gen);
    bool found = false;

    /* We can skip the initial node as we increased val. */
    node = avl_tree_next(node);
    while (node) {
        if (node == found_node) {
            found = true;
            break;
        }
        node = avl_tree_next(node);
    }

    /* These two are equivalent, it's just an assert. */
    if (!found || !node) {
        pal_printf("avl_tree_lower_bound has not found the next element!\n");
        DkProcessExit(1);
    }

    avl_tree_delete(&tree, &tmp.node);
    if (!debug_avl_tree_is_balanced(&tree)) {
        EXIT_UNBALANCED();
    }

    i = RAND_DEL_COUNT;
    while (i) {
        uint32_t r = rand() % ELEMENTS_COUNT;
        if (!t[r].freed) {
            t[r].freed = true;
            avl_tree_delete(&tree, &t[r].node);
            i--;
            if (!debug_avl_tree_is_balanced(&tree)) {
                EXIT_UNBALANCED();
            }
        }
    }
    for (i = 0; i < ELEMENTS_COUNT; i++) {
        if (!t[i].freed) {
            avl_tree_delete(&tree, &t[i].node);
            t[i].freed = true;
            if (!debug_avl_tree_is_balanced(&tree)) {
                EXIT_UNBALANCED();
            }
        }
    }
}

#define DIFF_ELEMENTS 0x10
#define _STR(x) #x
static void test_ordering(void) {
    static_assert(ELEMENTS_COUNT > DIFF_ELEMENTS,
                  "This test requires more than " _STR(DIFF_ELEMENTS) " elements!");
    ssize_t i;

    /* The newer node will be on the left, so we need to insert them in reverse order. */
    for (i = ELEMENTS_COUNT - 1; i >= 0; i--) {
        t[i].key = i / (ELEMENTS_COUNT / DIFF_ELEMENTS);
        avl_tree_insert(&tree, &t[i].node);
        if (!debug_avl_tree_is_balanced(&tree)) {
            EXIT_UNBALANCED();
        }
    }

    struct avl_tree_node* prev = avl_tree_first(&tree);
    struct avl_tree_node* node = avl_tree_next(prev);

    while (node) {
        /* These nodes are all a part of array `t`. */
        if ((uintptr_t)prev >= (uintptr_t)node) {
            pal_printf("Wrong ordering of nodes: %p %p\n", prev, node);
            DkProcessExit(1);
        }
        prev = node;
        node = avl_tree_next(prev);
    }

    for (i = 0; i < ELEMENTS_COUNT; i++) {
        avl_tree_delete(&tree, &t[i].node);
        if (!debug_avl_tree_is_balanced(&tree)) {
            EXIT_UNBALANCED();
        }
    }
}

static int32_t rand_mod(void) {
    return rand() % (ELEMENTS_COUNT / 4);
}

int main(void) {
    pal_printf("Running static tests: ");
    test_ordering();
    srand(1337);
    do_test(rand_mod);
    do_test(rand);
    pal_printf("Done!\n");

    uint32_t seed = 0;
    if (DkRandomBitsRead(&seed, sizeof(seed)) < 0) {
        pal_printf("Getting a seed failed\n");
        return 1;
    }
    pal_printf("Running dynamic tests (with seed: %u): ", seed);
    srand(seed);
    do_test(rand_mod);
    do_test(rand);
    pal_printf("Done!\n");

    return 0;
}
