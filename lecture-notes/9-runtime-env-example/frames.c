#include <stdio.h>
#include <stdint.h>
static void show(const char *fn, int *local) {
    uintptr_t *fp = __builtin_frame_address(1);   /* frame pointer (rbp) of the caller */
    printf("%-10s rbp=%p  saved rbp=[rbp]=%p  return addr=[rbp+8]=%p  local at %p (rbp%+ld)\n",
           fn, (void*)fp, (void*)fp[0], (void*)fp[1], (void*)local,
           (long)((char*)local - (char*)fp));
}
__attribute__((noinline)) int multiply(int a, int b) {
    int result = a * b;
    show("multiply", &result);
    return result;
}
__attribute__((noinline)) int calculate(void) {
    int x = 5;
    int y = multiply(x, 3);
    show("calculate", &y);
    return y + 1;
}
int main(void) {
    int final = calculate();
    show("main", &final);
    return final == 16 ? 0 : 1;
}
