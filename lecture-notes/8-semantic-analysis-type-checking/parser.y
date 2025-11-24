%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ast.h"
#include "symbol.h"
#include "type.h"

extern int line_number;
extern int yylex();
extern FILE* yyin;
void yyerror(const char* s);

// External declarations
extern SymbolTable* global_symtab;
extern struct ASTNode* program_ast;

// External type variables
extern Type* TYPE_INTEGER;
extern Type* TYPE_FLOATING;
extern Type* TYPE_BOOLEAN;
extern Type* TYPE_ERROR_TYPE;

Type* get_type_from_token(int token);

%}

%union {
    int int_val;
    double float_val;
    char* string_val;
    struct ASTNode* node;
    Type* type;
}

%token TOK_INT  TOK_BOOL
%token TOK_TRUE TOK_FALSE
%token TOK_PLUS TOK_MINUS TOK_MULTIPLY TOK_DIVIDE
%token TOK_ASSIGN TOK_SEMICOLON
%token TOK_LPAREN TOK_RPAREN
%token TOK_EOF
%token TOK_ERROR

%token <int_val> TOK_INTEGER
%token <float_val> TOK_FLOAT
%token <string_val> TOK_ID

%type <node> program statement_list statement expression term factor declaration assignment
%type <type> type_specifier

%%

program:
    statement_list TOK_EOF { 
        program_ast = $1;
        YYACCEPT;
    }
    ;

statement_list:
    statement {
        // Create a program node with single statement
        struct ASTNode** stmts = malloc(sizeof(struct ASTNode*));
        stmts[0] = $1;
        $$ = create_program(stmts, 1);
    }
    | statement_list statement {
        // Append statement to existing program
        int old_count = $1->data.program.statement_count;
        int new_count = old_count + 1;
        struct ASTNode** new_stmts = realloc($1->data.program.statements, 
                                           new_count * sizeof(struct ASTNode*));
        new_stmts[old_count] = $2;
        $1->data.program.statements = new_stmts;
        $1->data.program.statement_count = new_count;
        $$ = $1;
    }
    ;

statement:
    declaration TOK_SEMICOLON { $$ = $1; }
    | assignment TOK_SEMICOLON { $$ = $1; }
    | expression TOK_SEMICOLON { $$ = $1; }
    ;

declaration:
    type_specifier TOK_ID {
        Symbol* existing = lookup_symbol_current_scope(global_symtab, $2);
        if (existing) {
            printf("Error at line %d: Variable '%s' already declared\n", 
                   line_number, $2);
        } else {
            add_symbol(global_symtab, $2, $1, line_number);
        }
        $$ = create_declaration($2, type_to_string($1), line_number);
        $$->result_type = $1;
        free($2); // Free the string allocated by lexer
    }
    ;

type_specifier:
    TOK_INT { $$ = TYPE_INTEGER; }
    | TOK_FLOAT { $$ = TYPE_FLOATING; }
    | TOK_BOOL { $$ = TYPE_BOOLEAN; }
    ;

assignment:
    TOK_ID TOK_ASSIGN expression {
        Symbol* sym = lookup_symbol(global_symtab, $1);
        if (!sym) {
            printf("Error at line %d: Variable '%s' not declared\n", 
                   line_number, $1);
        } else if (!is_assignable(sym->type, $3->result_type, line_number)) {
            // Error message printed in is_assignable
        } else {
            sym->is_initialized = 1;
        }
        $$ = create_assignment($1, $3, line_number);
        $$->result_type = sym ? sym->type : TYPE_ERROR_TYPE;
        free($1); // Free the string allocated by lexer
    }
    ;

expression:
    expression TOK_PLUS term {
        Type* result_type = get_expression_type($1->result_type, $3->result_type, "+", line_number);
        $$ = create_binary_op(OP_ADD, $1, $3, line_number);
        $$->result_type = result_type;
    }
    | expression TOK_MINUS term {
        Type* result_type = get_expression_type($1->result_type, $3->result_type, "-", line_number);
        $$ = create_binary_op(OP_SUBTRACT, $1, $3, line_number);
        $$->result_type = result_type;
    }
    | term { $$ = $1; }
    ;

term:
    term TOK_MULTIPLY factor {
        Type* result_type = get_expression_type($1->result_type, $3->result_type, "*", line_number);
        $$ = create_binary_op(OP_MULTIPLY, $1, $3, line_number);
        $$->result_type = result_type;
    }
    | term TOK_DIVIDE factor {
        Type* result_type = get_expression_type($1->result_type, $3->result_type, "/", line_number);
        $$ = create_binary_op(OP_DIVIDE, $1, $3, line_number);
        $$->result_type = result_type;
    }
    | factor { $$ = $1; }
    ;

factor:
    TOK_INTEGER { 
        $$ = create_number($1, line_number);
        $$->result_type = TYPE_INTEGER;
    }
    | TOK_FLOAT { 
        // Create a float node (you might want to extend AST to support floats)
        $$ = create_number((int)$1, line_number); // Simplified for demo
        $$->result_type = TYPE_FLOATING;
    }
    | TOK_ID {
        Symbol* sym = lookup_symbol(global_symtab, $1);
        $$ = create_variable($1, line_number);
        if (!sym) {
            printf("Error at line %d: Variable '%s' not declared\n", 
                   line_number, $1);
            $$->result_type = TYPE_ERROR_TYPE;
        } else if (!sym->is_initialized) {
            printf("Warning at line %d: Variable '%s' might be uninitialized\n", 
                   line_number, $1);
            $$->result_type = sym->type;
        } else {
            $$->result_type = sym->type;
        }
        free($1); // Free the string allocated by lexer
    }
    | TOK_LPAREN expression TOK_RPAREN { 
        $$ = $2; 
    }
    | TOK_TRUE {
        $$ = create_number(1, line_number); // Represent true as 1
        $$->result_type = TYPE_BOOLEAN;
    }
    | TOK_FALSE {
        $$ = create_number(0, line_number); // Represent false as 0
        $$->result_type = TYPE_BOOLEAN;
    }
    ;

%%

void yyerror(const char* s) {
    fprintf(stderr, "Parser error at line %d: %s\n", line_number, s);
}

Type* get_type_from_token(int token) {
    switch(token) {
        case TOK_INT: return TYPE_INTEGER;
        case TOK_FLOAT: return TYPE_FLOATING;
        case TOK_BOOL: return TYPE_BOOLEAN;
        default: return TYPE_ERROR_TYPE;
    }
}