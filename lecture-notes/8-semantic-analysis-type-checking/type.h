#ifndef TYPE_H
#define TYPE_H

typedef enum {
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_BOOL,
    TYPE_VOID,
    TYPE_ERROR
} TypeKind;

typedef struct Type {
    TypeKind kind;
    char* name;
} Type;

Type* create_type(TypeKind kind);
Type* create_named_type(char* name);
int types_equal(Type* t1, Type* t2);
Type* get_expression_type(Type* left, Type* right, char* op, int line);
char* type_to_string(Type* type);

extern Type* TYPE_INTEGER;
extern Type* TYPE_FLOATING;
extern Type* TYPE_BOOLEAN;
extern Type* TYPE_VOID_TYPE;
extern Type* TYPE_ERROR_TYPE;

#endif