#ifndef CODEGEN_H
#define CODEGEN_H

#include "ast.h"

// Bytecode instruction set
typedef enum {
    OP_PUSH,    // Push number to stack
    OP_ADD,     // Add top two stack values
    OP_SUB,     // Subtract top two stack values  
    OP_MUL,     // Multiply top two stack values
    OP_DIV,     // Divide top two stack values
    OP_NEG,     // Negate top stack value
    OP_POW      // Power (top-1)^top
} OpCode;

// Single instruction
typedef struct {
    OpCode opcode;
    double operand;  // Used only for OP_PUSH
} Instruction;

// Complete program with instructions
typedef struct {
    Instruction *code;
    int size;
    int capacity;
} Bytecode;

// Code generation interface
Bytecode *codegen_generate(ASTNode *ast);
void codegen_execute(Bytecode *bytecode);
void codegen_disassemble(Bytecode *bytecode);
void codegen_free(Bytecode *bytecode);

#endif