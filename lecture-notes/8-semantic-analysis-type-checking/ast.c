#include "ast.h"
#include "type.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct ASTNode* create_number(int value, int line) {
    struct ASTNode* node = malloc(sizeof(struct ASTNode));
    node->type = NODE_NUMBER;
    node->line_number = line;
    node->result_type = NULL;
    node->data.number.value = value;
    return node;
}

struct ASTNode* create_variable(char* name, int line) {
    struct ASTNode* node = malloc(sizeof(struct ASTNode));
    node->type = NODE_VARIABLE;
    node->line_number = line;
    node->result_type = NULL;
    node->data.variable.name = strdup(name);
    return node;
}

struct ASTNode* create_binary_op(Operator op, struct ASTNode* left, 
                                struct ASTNode* right, int line) {
    struct ASTNode* node = malloc(sizeof(struct ASTNode));
    node->type = NODE_BINARY_OP;
    node->line_number = line;
    node->result_type = NULL;
    node->data.binary_op.op = op;
    node->data.binary_op.left = left;
    node->data.binary_op.right = right;
    return node;
}

struct ASTNode* create_assignment(char* var_name, struct ASTNode* value, int line) {
    struct ASTNode* node = malloc(sizeof(struct ASTNode));
    node->type = NODE_ASSIGNMENT;
    node->line_number = line;
    node->result_type = NULL;
    node->data.assignment.var_name = strdup(var_name);
    node->data.assignment.value = value;
    return node;
}

struct ASTNode* create_declaration(char* var_name, char* type_name, int line) {
    struct ASTNode* node = malloc(sizeof(struct ASTNode));
    node->type = NODE_DECLARATION;
    node->line_number = line;
    node->result_type = NULL;
    node->data.declaration.var_name = strdup(var_name);
    node->data.declaration.type_name = strdup(type_name);
    return node;
}

struct ASTNode* create_program(struct ASTNode** statements, int count) {
    struct ASTNode* node = malloc(sizeof(struct ASTNode));
    node->type = NODE_PROGRAM;
    node->line_number = 0;
    node->result_type = NULL;
    node->data.program.statements = statements;
    node->data.program.statement_count = count;
    return node;
}

void print_ast(struct ASTNode* node, int indent) {
    if (!node) return;
    
    for (int i = 0; i < indent; i++) printf("  ");
    
    switch (node->type) {
        case NODE_NUMBER:
            printf("Number: %d", node->data.number.value);
            if (node->result_type) {
                printf(" [type: %s]", type_to_string(node->result_type));
            }
            printf("\n");
            break;
        case NODE_VARIABLE:
            printf("Variable: %s", node->data.variable.name);
            if (node->result_type) {
                printf(" [type: %s]", type_to_string(node->result_type));
            }
            printf("\n");
            break;
        case NODE_BINARY_OP: {
            char* op_str = "?";
            switch (node->data.binary_op.op) {
                case OP_ADD: op_str = "+"; break;
                case OP_SUBTRACT: op_str = "-"; break;
                case OP_MULTIPLY: op_str = "*"; break;
                case OP_DIVIDE: op_str = "/"; break;
            }
            printf("BinaryOp: %s", op_str);
            if (node->result_type) {
                printf(" [type: %s]", type_to_string(node->result_type));
            }
            printf("\n");
            print_ast(node->data.binary_op.left, indent + 1);
            print_ast(node->data.binary_op.right, indent + 1);
            break;
        }
        case NODE_ASSIGNMENT:
            printf("Assignment: %s", node->data.assignment.var_name);
            if (node->result_type) {
                printf(" [type: %s]", type_to_string(node->result_type));
            }
            printf("\n");
            print_ast(node->data.assignment.value, indent + 1);
            break;
        case NODE_DECLARATION:
            printf("Declaration: %s %s", node->data.declaration.type_name, 
                   node->data.declaration.var_name);
            if (node->result_type) {
                printf(" [type: %s]", type_to_string(node->result_type));
            }
            printf("\n");
            break;
        case NODE_PROGRAM:
            printf("Program (%d statements):\n", node->data.program.statement_count);
            for (int i = 0; i < node->data.program.statement_count; i++) {
                print_ast(node->data.program.statements[i], indent + 1);
            }
            break;
    }
}

void free_ast(struct ASTNode* node) {
    if (!node) return;
    
    switch (node->type) {
        case NODE_VARIABLE:
            free(node->data.variable.name);
            break;
        case NODE_BINARY_OP:
            free_ast(node->data.binary_op.left);
            free_ast(node->data.binary_op.right);
            break;
        case NODE_ASSIGNMENT:
            free(node->data.assignment.var_name);
            free_ast(node->data.assignment.value);
            break;
        case NODE_DECLARATION:
            free(node->data.declaration.var_name);
            free(node->data.declaration.type_name);
            break;
        case NODE_PROGRAM:
            for (int i = 0; i < node->data.program.statement_count; i++) {
                free_ast(node->data.program.statements[i]);
            }
            free(node->data.program.statements);
            break;
        case NODE_NUMBER:
            // Nothing to free for number nodes
            break;
    }
    free(node);
}