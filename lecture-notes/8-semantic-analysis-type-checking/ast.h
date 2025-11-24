#ifndef AST_H
#define AST_H

// Forward declaration to avoid circular dependency
typedef struct Type Type;

typedef enum {
    NODE_NUMBER,
    NODE_VARIABLE,
    NODE_BINARY_OP,
    NODE_ASSIGNMENT,
    NODE_DECLARATION,
    NODE_PROGRAM
} NodeType;

typedef enum {
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE
} Operator;

struct ASTNode {
    NodeType type;
    int line_number;
    Type* result_type;
    union {
        struct {
            int value;
        } number;
        
        struct {
            char* name;
        } variable;
        
        struct {
            Operator op;
            struct ASTNode* left;
            struct ASTNode* right;
        } binary_op;
        
        struct {
            char* var_name;
            struct ASTNode* value;
        } assignment;
        
        struct {
            char* var_name;
            char* type_name;
        } declaration;
        
        struct {
            struct ASTNode** statements;
            int statement_count;
        } program;
    } data;
};

struct ASTNode* create_number(int value, int line);
struct ASTNode* create_variable(char* name, int line);
struct ASTNode* create_binary_op(Operator op, struct ASTNode* left, 
                                struct ASTNode* right, int line);
struct ASTNode* create_assignment(char* var_name, struct ASTNode* value, int line);
struct ASTNode* create_declaration(char* var_name, char* type_name, int line);
struct ASTNode* create_program(struct ASTNode** statements, int count);

void print_ast(struct ASTNode* node, int indent);
void free_ast(struct ASTNode* node);

#endif