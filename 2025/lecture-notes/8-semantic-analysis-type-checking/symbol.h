#ifndef SYMBOL_H
#define SYMBOL_H

#include "type.h"

typedef struct Symbol {
    char* name;
    Type* type;
    int line_declared;
    int is_initialized;
    struct Symbol* next;
} Symbol;

typedef struct SymbolTable {
    Symbol* symbols;
    struct SymbolTable* parent;
    int scope_level;
} SymbolTable;

SymbolTable* create_symbol_table(SymbolTable* parent);
void destroy_symbol_table(SymbolTable* table);
Symbol* add_symbol(SymbolTable* table, char* name, Type* type, int line);
Symbol* lookup_symbol(SymbolTable* table, char* name);
Symbol* lookup_symbol_current_scope(SymbolTable* table, char* name);
int is_assignable(Type* target, Type* source, int line);

#endif