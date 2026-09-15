#include <stdio.h>
#include <stdlib.h>
#include "ast.h"
#include "symbol.h"
#include "type.h"

// Global variables - define here, declare extern in other files
SymbolTable* global_symtab = NULL;
struct ASTNode* program_ast = NULL;

// External lexer/parser functions
extern FILE* yyin;
extern int yyparse();

// Type variables - defined in type.c
extern Type* TYPE_INTEGER;
extern Type* TYPE_FLOATING;
extern Type* TYPE_BOOLEAN;
extern Type* TYPE_VOID_TYPE;
extern Type* TYPE_ERROR_TYPE;

void initialize_types() {
    TYPE_INTEGER = create_type(TYPE_INT);
    TYPE_FLOATING = create_type(TYPE_FLOAT);
    TYPE_BOOLEAN = create_type(TYPE_BOOL);
    TYPE_VOID_TYPE = create_type(TYPE_VOID);
    TYPE_ERROR_TYPE = create_type(TYPE_ERROR);
}

int main(int argc, char** argv) {
    // Initialize type system
    initialize_types();
    
    // Create global symbol table
    global_symtab = create_symbol_table(NULL);
    
    if (argc < 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }
    
    // Open input file
    FILE* input_file = fopen(argv[1], "r");
    if (!input_file) {
        printf("Error: Cannot open file %s\n", argv[1]);
        return 1;
    }
    
    // Set flex to read from file
    yyin = input_file;
    
    printf("=== Parsing and Semantic Analysis ===\n");
    
    // Run parser
    if (yyparse() == 0) {
        printf("Parse successful!\n");
        
        if (program_ast) {
            printf("\n=== Abstract Syntax Tree ===\n");
            print_ast(program_ast, 0);
        }
    } else {
        printf("Parse failed with errors.\n");
    }
    
    // Cleanup
    if (program_ast) {
        free_ast(program_ast);
    }
    destroy_symbol_table(global_symtab);
    fclose(input_file);
    
    return 0;
}