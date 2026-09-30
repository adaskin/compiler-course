#include "type.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Define the global type variables
Type* TYPE_INTEGER = NULL;
Type* TYPE_FLOATING = NULL;
Type* TYPE_BOOLEAN = NULL;
Type* TYPE_VOID_TYPE = NULL;
Type* TYPE_ERROR_TYPE = NULL;

Type* create_type(TypeKind kind) {
    Type* type = malloc(sizeof(Type));
    type->kind = kind;
    
    switch (kind) {
        case TYPE_INT: type->name = strdup("int"); break;
        case TYPE_FLOAT: type->name = strdup("float"); break;
        case TYPE_BOOL: type->name = strdup("bool"); break;
        case TYPE_VOID: type->name = strdup("void"); break;
        case TYPE_ERROR: type->name = strdup("error"); break;
    }
    return type;
}

Type* create_named_type(char* name) {
    Type* type = malloc(sizeof(Type));
    type->name = strdup(name);
    // Default to int for simplicity
    type->kind = TYPE_INT;
    return type;
}

int types_equal(Type* t1, Type* t2) {
    if (!t1 || !t2) return 0;
    return t1->kind == t2->kind;
}

Type* get_expression_type(Type* left, Type* right, char* op, int line) {
    if (!left || !right) return TYPE_ERROR_TYPE;
    
    // For now, simple type checking - both operands must be same type
    if (types_equal(left, right)) {
        return left;
    }
    
    printf("Error at line %d: Type mismatch in %s operation\n", line, op);
    return TYPE_ERROR_TYPE;
}

char* type_to_string(Type* type) {
    if (!type) return "unknown";
    return type->name;
}