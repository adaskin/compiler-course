#include "codegen.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INITIAL_CAPACITY 16

// Create new bytecode container
Bytecode *codegen_create(void) {
    Bytecode *bc = malloc(sizeof(Bytecode));
    bc->code = malloc(sizeof(Instruction) * INITIAL_CAPACITY);
    bc->size = 0;
    bc->capacity = INITIAL_CAPACITY;
    return bc;
}

// Add instruction to bytecode
void codegen_emit(Bytecode *bc, OpCode opcode, double operand) {
    // Resize if needed
    if (bc->size >= bc->capacity) {
        bc->capacity *= 2;
        bc->code = realloc(bc->code, sizeof(Instruction) * bc->capacity);
    }
    
    // Add instruction
    bc->code[bc->size].opcode = opcode;
    bc->code[bc->size].operand = operand;
    bc->size++;
}

// Recursive code generation from AST
void codegen_generate_from_ast(Bytecode *bc, ASTNode *node) {
    if (!node) return;
    
    switch (node->type) {
        case AST_NUMBER:
            codegen_emit(bc, OP_PUSH, node->value);
            break;
            
        case AST_ADD:
            codegen_generate_from_ast(bc, node->left);
            codegen_generate_from_ast(bc, node->right);
            codegen_emit(bc, OP_ADD, 0);
            break;
            
        case AST_SUB:
            codegen_generate_from_ast(bc, node->left);
            codegen_generate_from_ast(bc, node->right);
            codegen_emit(bc, OP_SUB, 0);
            break;
            
        case AST_MUL:
            codegen_generate_from_ast(bc, node->left);
            codegen_generate_from_ast(bc, node->right);
            codegen_emit(bc, OP_MUL, 0);
            break;
            
        case AST_DIV:
            codegen_generate_from_ast(bc, node->left);
            codegen_generate_from_ast(bc, node->right);
            codegen_emit(bc, OP_DIV, 0);
            break;
            
        case AST_NEG:
            codegen_generate_from_ast(bc, node->child);
            codegen_emit(bc, OP_NEG, 0);
            break;
            
        case AST_POW:
            codegen_generate_from_ast(bc, node->left);
            codegen_generate_from_ast(bc, node->right);
            codegen_emit(bc, OP_POW, 0);
            break;
    }
}

// Generate bytecode from AST
Bytecode *codegen_generate(ASTNode *ast) {
    Bytecode *bc = codegen_create();
    codegen_generate_from_ast(bc, ast);
    return bc;
}

// Execute bytecode on stack machine
void codegen_execute(Bytecode *bytecode) {
    double stack[256];
    int sp = -1;  // Stack pointer
    
    for (int ip = 0; ip < bytecode->size; ip++) {
        Instruction instr = bytecode->code[ip];
        
        switch (instr.opcode) {
            case OP_PUSH:
                stack[++sp] = instr.operand;
                break;
                
            case OP_ADD:
                stack[sp-1] = stack[sp-1] + stack[sp];
                sp--;
                break;
                
            case OP_SUB:
                stack[sp-1] = stack[sp-1] - stack[sp];
                sp--;
                break;
                
            case OP_MUL:
                stack[sp-1] = stack[sp-1] * stack[sp];
                sp--;
                break;
                
            case OP_DIV:
                stack[sp-1] = stack[sp-1] / stack[sp];
                sp--;
                break;
                
            case OP_NEG:
                stack[sp] = -stack[sp];
                break;
                
            case OP_POW:
                stack[sp-1] = pow(stack[sp-1], stack[sp]);
                sp--;
                break;
        }
    }
    
    if (sp == 0) {
        printf("Bytecode execution result: %g\n", stack[0]);
    } else {
        printf("Error: Stack imbalance (sp=%d)\n", sp);
    }
}

// Disassemble bytecode for debugging
void codegen_disassemble(Bytecode *bytecode) {
    printf("Bytecode disassembly (%d instructions):\n", bytecode->size);
    for (int i = 0; i < bytecode->size; i++) {
        printf("  %3d: ", i);
        Instruction instr = bytecode->code[i];
        
        switch (instr.opcode) {
            case OP_PUSH: printf("PUSH %g\n", instr.operand); break;
            case OP_ADD:  printf("ADD\n"); break;
            case OP_SUB:  printf("SUB\n"); break;
            case OP_MUL:  printf("MUL\n"); break;
            case OP_DIV:  printf("DIV\n"); break;
            case OP_NEG:  printf("NEG\n"); break;
            case OP_POW:  printf("POW\n"); break;
        }
    }
}

// Free bytecode memory
void codegen_free(Bytecode *bytecode) {
    if (bytecode) {
        free(bytecode->code);
        free(bytecode);
    }
}