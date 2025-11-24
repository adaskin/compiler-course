/*https://www3.nd.edu/~dthain/compilerbook/chapter3.pdf*/
typedef enum {
    TOKEN_EOF=0,
    TOKEN_WHILE,
    TOKEN_ADD,
    TOKEN_IDENT,
    TOKEN_NUMBER,
    TOKEN_ERROR
} token_t;