#include "ast.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

ASTNode *ast_create_number(double value) {
    ASTNode *node = malloc(sizeof(ASTNode));
    node->type = AST_NUMBER;
    node->value = value;
    node->left = node->right = node->child = NULL;
    return node;
}

ASTNode *ast_create_binary(ASTNodeType type, ASTNode *left, ASTNode *right) {
    ASTNode *node = malloc(sizeof(ASTNode));
    node->type = type;
    node->left = left;
    node->right = right;
    node->child = NULL;
    return node;
}

ASTNode *ast_create_unary(ASTNodeType type, ASTNode *child) {
    ASTNode *node = malloc(sizeof(ASTNode));
    node->type = type;
    node->child = child;
    node->left = node->right = NULL;
    return node;
}

double ast_evaluate(ASTNode *node) {
    if (!node) return 0;
    
    switch (node->type) {
        case AST_NUMBER: return node->value;
        case AST_ADD: return ast_evaluate(node->left) + ast_evaluate(node->right);
        case AST_SUB: return ast_evaluate(node->left) - ast_evaluate(node->right);
        case AST_MUL: return ast_evaluate(node->left) * ast_evaluate(node->right);
        case AST_DIV: return ast_evaluate(node->left) / ast_evaluate(node->right);
        case AST_NEG: return -ast_evaluate(node->child);
        case AST_POW: return pow(ast_evaluate(node->left), ast_evaluate(node->right));
        default: return 0;
    }
}

void ast_free(ASTNode *node) {
    if (!node) return;
    ast_free(node->left);
    ast_free(node->right);
    ast_free(node->child);
    free(node);
}

void ast_print(ASTNode *node, int depth) {
    if (!node) return;
    for (int i = 0; i < depth; i++) printf("  ");
    switch (node->type) {
        case AST_NUMBER: printf("NUMBER: %g\n", node->value); break;
        case AST_ADD: printf("ADD\n"); break;
        case AST_SUB: printf("SUB\n"); break;
        case AST_MUL: printf("MUL\n"); break;
        case AST_DIV: printf("DIV\n"); break;
        case AST_NEG: printf("NEG\n"); break;
        case AST_POW: printf("POW\n"); break;
    }
    ast_print(node->left, depth + 1);
    ast_print(node->right, depth + 1);
    ast_print(node->child, depth + 1);
}