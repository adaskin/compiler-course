# Runtime environment examples (lecture 9)

All examples were run with gcc 13 on x86-64 Linux.

| File | What to try |
|------|-------------|
| `calc.c` | `gcc -O0 -S -masm=intel calc.c` then `-O2`: where did the frame go? |
| `frames.c` | `gcc -O0 -fno-omit-frame-pointer -fno-stack-protector -o frames frames.c && ./frames` prints the control-link chain of three frames |
| `fact.c` | `gcc -O2 -S -masm=intel fact.c`: both `fact` and `fact_acc` become loops (an optimization, not a guarantee) |
| `rec.py` | Python's recursion limit (1000) and `RecursionError` |
| `uaf.c`, `leak.c` | `gcc -g -fsanitize=address uaf.c && ./a.out`: AddressSanitizer / LeakSanitizer reports |
| `sz.c` | `sizeof(struct Point)` is 16 on a 64-bit machine |
