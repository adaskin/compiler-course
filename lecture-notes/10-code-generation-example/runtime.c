/* Minimal runtime for the mini functional language: print built-ins and the
   division-by-zero handler. Linked with:  clang prog.ll runtime.c -o prog   */
#include <stdio.h>
#include <stdlib.h>

int print_int(int x)  { printf("%d\n", x); return 0; }
int print_bool(int b) { printf("%s\n", b ? "true" : "false"); return 0; }

void fpl_div_zero(void) {
    fprintf(stderr, "runtime error: division by zero\n");
    exit(1);
}
