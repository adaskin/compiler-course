# Code generation example (lecture 10)

A **small, readable** compiler for the course's example language (see `final-project.md`, Section 3):
`fplc_mini.py` = lexer + parser + type checker + LLVM IR generator, about 270 lines of Python.
It is for reading next to the lecture. It is **not a reference solution**: no error recovery,
diagnostics are Python exceptions, and it supports only what the lecture shows.

```bash
python3 fplc_mini.py tests/example.fpl > example.ll     # emit LLVM IR (opaque pointers, LLVM 15+)
opt -passes=verify -disable-output example.ll           # verify
clang example.ll runtime.c -o example && ./example      # build and run: prints 10
./run_tests.sh                                          # all tests, three ways (see below)
```

Tested with Ubuntu clang/LLVM 18.1 and Python 3.13.

| File | Purpose |
|------|---------|
| `fplc_mini.py` | the compiler (typed AST -> textual LLVM IR) |
| `interp.py` | reference interpreter over the same typed AST: the **oracle** for differential testing |
| `runtime.c` | `print_int`, `print_bool`, `fpl_div_zero` |
| `run_tests.sh` | each test compiled at `-O0` and `-O2` and interpreted; every `.ll` must pass the verifier |
| `tests/*.fpl`, `*.expected`, `*.exit` | 9 programs with hand-computed output (and expected exit code) |
| `mips_cgen.py`, `mips_sim.py` | the stack-machine / MIPS generator of lecture 10 part 2, and a tiny simulator used only to test it (real tools: SPIM, MARS); the generated assembly also assembles with `llvm-mc -triple=mips-linux-gnu` |
| `c_snippet/emit.c` | the same `let`/`if`/`phi` rules in C (path A) |
| `musttail_count.ll` | a 100-million-deep recursion that runs in constant stack at `-O0` thanks to `musttail` |

## Language restrictions of this mini compiler

- Top-level values are visible to the **later top-level items and the final expression**, not inside function bodies.
- Lambdas see only top-level functions (no capture), as in the project.
- Function types are written `int -> int`; built-ins `print_int` and `print_bool` cannot be passed as values.

## A parser note (layout rule)

With application by juxtaposition (`app = atom { atom }`), the project's example

```
let answer = ... else fact 3

print_int answer
```

is ambiguous: `fact 3 print_int answer` is also a valid application. This mini compiler resolves it with a
**layout rule**: a token in column 0 starts the next top-level item, so it cannot continue an application.
Your own SPEC must pick some rule (layout, a terminator token, or requiring `in` for the final expression).
