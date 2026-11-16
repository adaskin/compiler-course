#include <stdio.h>
struct Point { int x, y; struct Point *next; };
int main(void){ printf("%zu\n", sizeof(struct Point)); }
