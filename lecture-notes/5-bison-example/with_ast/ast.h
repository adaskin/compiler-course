#ifndef AST_H
#define AST_H

typedef enum {
    AST_NUMBER,
    AST_ADD,
    AST_SUB,
    AST_MUL,
    AST_DIV,
    AST_NEG,
    AST_POW
} ASTNodeType;

typedef struct ASTNode {
    ASTNodeType type;
    double value;
    struct ASTNode *left;
    struct ASTNode *right;
    struct ASTNode *child;
} ASTNode;

ASTNode *ast_create_number(double value);
ASTNode *ast_create_binary(ASTNodeType type, ASTNode *left, ASTNode *right);
ASTNode *ast_create_unary(ASTNodeType type, ASTNode *child);
double ast_evaluate(ASTNode *node);
void ast_free(ASTNode *node);
void ast_print(ASTNode *node, int depth);

#endif