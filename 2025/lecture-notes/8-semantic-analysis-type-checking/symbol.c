#include "symbol.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

SymbolTable* create_symbol_table(SymbolTable* parent) {
    SymbolTable* table = malloc(sizeof(SymbolTable));
    table->symbols = NULL;
    table->parent = parent;
    table->scope_level = parent ? parent->scope_level + 1 : 0;
    return table;
}

void destroy_symbol_table(SymbolTable* table) {
    Symbol* current = table->symbols;
    while (current) {
        Symbol* next = current->next;
        free(current->name);
        free(current);
        current = next;
    }
    free(table);
}

Symbol* add_symbol(SymbolTable* table, char* name, Type* type, int line) {
    Symbol* symbol = malloc(sizeof(Symbol));
    symbol->name = strdup(name);
    symbol->type = type;
    symbol->line_declared = line;
    symbol->is_initialized = 0;
    symbol->next = table->symbols;
    table->symbols = symbol;
    return symbol;
}

Symbol* lookup_symbol(SymbolTable* table, char* name) {
    SymbolTable* current_table = table;
    while (current_table) {
        Symbol* current = current_table->symbols;
        while (current) {
            if (strcmp(current->name, name) == 0) {
                return current;
            }
            current = current->next;
        }
        current_table = current_table->parent;
    }
    return NULL;
}

Symbol* lookup_symbol_current_scope(SymbolTable* table, char* name) {
    Symbol* current = table->symbols;
    while (current) {
        if (strcmp(current->name, name) == 0) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

int is_assignable(Type* target, Type* source, int line) {
    if (!target || !source) return 0;
    
    if (types_equal(target, source)) {
        return 1;
    }
    
    // Allow int to float conversion
    if (target->kind == TYPE_FLOAT && source->kind == TYPE_INT) {
        return 1;
    }
    
    printf("Error at line %d: Cannot assign %s to %s\n", 
           line, type_to_string(source), type_to_string(target));
    return 0;
}