/* Path A (C): emitting textual LLVM IR from an AST. Expression kinds: int, var, +,-,*, <, if, let.
   Build:  gcc -o emit emit.c && ./emit > f.ll && clang f.ll driver.c -o f && ./f   */
#include <stdio.h>
#include <string.h>

typedef enum { N_INT, N_VAR, N_BIN, N_IF, N_LET } Kind;
typedef struct Node {
    Kind kind;
    int value;                      /* N_INT */
    const char *name;               /* N_VAR, N_LET */
    char op;                        /* N_BIN: '+', '-', '*', '<' */
    struct Node *a, *b, *c;         /* operands / cond,then,else / init,body */
} Node;

typedef struct Env { const char *name; const char *val; struct Env *next; } Env;   /* name -> SSA operand */

static int counter = 0;                         /* fresh names for temporaries and labels */
static char cur[32] = "entry";                  /* label of the block we are emitting into */
static char pool[256][32]; static int pool_n = 0;
static char *fresh(const char *base) { snprintf(pool[pool_n], 32, "%s.%d", base, ++counter); return pool[pool_n++]; }
static void start(const char *label) { printf("%s:\n", label); strcpy(cur, label); }

static const char *lookup(Env *e, const char *n) { for (; e; e = e->next) if (!strcmp(e->name, n)) return e->val; return "?"; }

static const char *gen(Node *n, Env *env) {
    char num[32];
    switch (n->kind) {
    case N_INT: snprintf(num, 32, "%d", n->value); return strcpy(pool[pool_n++], num);
    case N_VAR: return lookup(env, n->name);
    case N_BIN: {
        const char *l = gen(n->a, env), *r = gen(n->b, env), *t = fresh("%t");
        if (n->op == '<') printf("  %s = icmp slt i32 %s, %s\n", t, l, r);
        else printf("  %s = %s i32 %s, %s\n", t, n->op == '+' ? "add" : n->op == '-' ? "sub" : "mul", l, r);
        return t;
    }
    case N_LET: {                                /* no instruction at all: just extend the environment */
        Env e = { n->name, gen(n->a, env), env };
        return gen(n->b, &e);
    }
    case N_IF: {
        const char *c = gen(n->a, env);
        char *th = fresh("then"), *el = fresh("else"), *jn = fresh("join");
        printf("  br i1 %s, label %%%s, label %%%s\n", c, th, el);
        char th_end[32], el_end[32];
        start(th); const char *x = gen(n->b, env); strcpy(th_end, cur); printf("  br label %%%s\n", jn);
        start(el); const char *y = gen(n->c, env); strcpy(el_end, cur); printf("  br label %%%s\n", jn);
        start(jn); char *t = fresh("%t");        /* the predecessors are the blocks we ended in */
        printf("  %s = phi i32 [ %s, %%%s ], [ %s, %%%s ]\n", t, x, th_end, y, el_end);
        return t;
    }
    }
    return "?";
}

int main(void) {
    /* let x = 5 in if x < 10 then (if x < 3 then 1 else x * 2) else 0 */
    Node five = { N_INT, 5 }, ten = { N_INT, 10 }, three = { N_INT, 3 }, one = { N_INT, 1 }, two = { N_INT, 2 }, zero = { N_INT, 0 };
    Node x = { N_VAR, 0, "x" };
    Node lt10 = { N_BIN, 0, 0, '<', &x, &ten }, lt3 = { N_BIN, 0, 0, '<', &x, &three }, dbl = { N_BIN, 0, 0, '*', &x, &two };
    Node inner = { N_IF, 0, 0, 0, &lt3, &one, &dbl };
    Node outer = { N_IF, 0, 0, 0, &lt10, &inner, &zero };
    Node prog = { N_LET, 0, "x", 0, &five, &outer };
    printf("define i32 @f() {\n"); start("entry");
    const char *r = gen(&prog, NULL);
    printf("  ret i32 %s\n}\n", r);
    return 0;
}
