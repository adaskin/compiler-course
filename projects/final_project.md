# BIL 463 — Final Project: A Functional Language That Humans and LLMs Can Both Read

**Fall 2026 · Ammar Daşkın**
**Groups:** 2–3 students · **One project, three phases(GRADED AS 2 PROJECTS)**
**Due:** End of Week 14 · **Demos:** Week 15

---

## 1. The Goal

Design a small **functional programming language** and build a compiler for it. Your compiler must:

1. Parse source code into an **AST**
2. **Type check** it with clear error messages
3. Generate **LLVM IR** that runs correctly
4. Be validated by tests showing correctness, errors, and recovery

You choose the name, syntax, and personality of your language. What you don't choose: it must type check and generate real code. That's the project.

**The theme:** your language should be easy for **humans to read** *and* easy for **LLMs to work with**. You'll design for both audiences, then measure how well an LLM handles your language.

One codebase, three phases. Each phase builds on the previous one and is graded when you reach it.

---

## 2. Language Requirements

### Must support

| Feature | Requirement |
|---------|-------------|
| Types | `int`, `bool`, and function types `T -> T` |
| Literals | Integers and `true`, `false` |
| Arithmetic | `+`, `-`, `*`, `/` on integers (and unary `-`) |
| Comparison | `<`, `<=`, `>`, `>=`, `==`, `!=` |
| Boolean operators | `&&`, `\|\|` (short-circuit) |
| Conditionals | `if e1 then e2 else e3` — an **expression**; both branches same type |
| Bindings | `let x = e1 in e2` — lexical scope, shadowing allowed |
| Functions | Named top-level functions with full type annotations, plus anonymous `fun (x: int) -> e` |
| Higher-order | Functions can be passed as arguments |
| Built-ins | `print_int : int -> int` and `print_bool : bool -> int` |

### Fixed semantics (so everyone's tests work the same way)

- `int` is 32-bit signed, wraps on overflow
- Evaluation is **strict, left-to-right, call-by-value**
- Functions are **not** partially applied — a call supplies all arguments
- Anonymous functions **may not capture** enclosing variables (reject with an error, or implement closures as a bonus)
- Division by zero: print to stderr, exit nonzero
- No mutation, no loops, no I/O beyond the print built-ins
- Any error = nonzero exit, no code generated. Error recovery reports multiple errors but **never accepts an invalid program**

### Type rules

| Construct | Rule |
|-----------|------|
| `if e1 then e2 else e3 : T` | `e1 : bool`, `e2 : T`, `e3 : T` |
| `let x = e1 in e2 : T` | `e1 : S`, `e2 : T` with `x : S` in scope |
| `fun (x: S) -> e : S -> T` | `e : T` with `x : S` in scope |
| `f a : T` | `f : S -> T` and `a : S` |

Function parameters must be annotated (no type inference). This is a deliberate design choice — explain it in your report.

---

## 3. Example

Here's one possible language. You can keep this, change it, or design something completely different.

```
let double (x: int): int = x * 2

let apply (f: int -> int) (x: int): int = f x

let fact (n: int): int =
  if n <= 1 then 1 else n * fact (n - 1)

let answer =
  let x = 5 in
  let y = apply double x in
  if y > fact 3 then y else fact 3

print_int answer
```

Output: `10`

### Grammar (fits on one screen)

```ebnf
program   = { def } expr ;
def       = "let" IDENT { "(" IDENT ":" type ")" } [ ":" type ] "=" expr ;
type      = atype [ "->" type ] ;
atype     = "int" | "bool" | "(" type ")" ;
expr      = "let" IDENT "=" expr "in" expr
          | "if" expr "then" expr "else" expr
          | "fun" "(" IDENT ":" type ")" "->" expr
          | or_expr ;
or_expr   = and_expr { "||" and_expr } ;
and_expr  = cmp_expr { "&&" cmp_expr } ;
cmp_expr  = add_expr [ ( "<" | "<=" | ">" | ">=" | "==" | "!=" ) add_expr ] ;
add_expr  = mul_expr { ( "+" | "-" ) mul_expr } ;
mul_expr  = unary { ( "*" | "/" ) unary } ;
unary     = "-" unary | app ;
app       = atom { atom } ;
atom      = INT | "true" | "false" | IDENT | "(" expr ")" ;
```

### What the LLVM output looks like

```llvm
define i32 @fact(i32 %n) {
entry:
  %c = icmp sle i32 %n, 1
  br i1 %c, label %then, label %else
then:
  br label %join
else:
  %n1 = sub i32 %n, 1
  %r = call i32 @fact(i32 %n1)
  %m = mul i32 %n, %r
  br label %join
join:
  %v = phi i32 [ 1, %then ], [ %m, %else ]
  ret i32 %v
}
```

Immutable `let` bindings become SSA values. Functions passed as arguments are `ptr` values. No closures needed for the baseline.

---

## 4. CLI and Diagnostics

**Fixed command-line interface** (so I can test everyone's compiler the same way):

```bash
./fplc prog.fpl --emit-ast         # dump AST
./fplc prog.fpl --emit-ll -o prog.ll   # emit LLVM IR
```

**Diagnostic format** (stderr):
```
<file>:<line>:<col>: error: <message>
```

Exit code `0` on success, `1` on any error. The compiler never crashes, hangs, or segfaults.

### Implementation paths

| Path | Tools |
|------|-------|
| A | C + Flex/Bison, emit `.ll` text |
| B | Python + PLY, emit `.ll` text or `llvmlite` |
| C | OCaml + ocamllex/Menhir, emit `.ll` text |

Emitting readable `.ll` text is encouraged. Build with `clang prog.ll runtime.c -o prog`. A minimal `runtime.c` provides the print built-ins.

---

## 5. Phases

### Phase 1: Specification, Parser, AST (30%)

- **SPEC.md** (max 2 pages): grammar, tokens, precedence, type rules, 5+ example programs
- Working lexer and parser
- **AST** (not a raw parse tree)
- `--emit-ast` flag
- Syntax errors with line/column
- **Panic-mode recovery**: resynchronize on `in`, `then`, `else`, `let`, `)`, or EOF; report multiple independent errors

### Phase 2: Type Checker (30%)

- Symbol table with lexical scoping and shadowing
- All type rules from Section 2
- Recursive and mutually recursive functions checked against their signatures
- Capture detection for lambdas (reject with clear error)
- Diagnostics with location, expected type, found type
- **Recovery**: report all independent errors in one run; use an `error` type internally so one mistake doesn't cascade

### Phase 3: LLVM Code Generation (40%)

- `--emit-ll` producing valid IR (passes `opt -verify`)
- Arithmetic → `add`/`sub`/`mul`/`sdiv`/`icmp`
- `if` → `br` + `phi`
- `let` → SSA values
- Functions, calls, recursion, function values
- Division-by-zero check
- Tiny C runtime for print built-ins
- One command (`run_tests.sh`) builds and runs everything

---

## 6. Tests

You write your own tests. Structure:

```
tests/
├── valid/       prog.fpl + prog.expected (stdout)
├── errors/      prog.fpl + prog.expected (diagnostics)
└── recovery/    prog.fpl + prog.expected (multiple errors reported)
```

| Category | Minimum | What it shows |
|----------|---------|---------------|
| **Correctness** | 10 | Precedence, shadowing, recursion, higher-order, `if` as expression |
| **Errors** | 5 syntax + 5 type | Wrong types, unbound variables, capture attempts, etc. |
| **Recovery** | 3 | Multiple errors in one file, no cascading |
| **Runtime** | 1 | Division by zero diagnosed |

**Rules:**
- `run_tests.sh` builds the compiler, runs all tests, prints PASS/FAIL
- Every `.ll` file passes `opt -verify`
- **Don't validate your compiler against itself.** Hand-compute expected outputs, or write a small reference interpreter (~50 lines, strongly recommended)
- At least one 30+ line program showing the language is usable (primes, fibonacci, etc.)
- Truncating any valid test at a random point must not crash the compiler

---

## 7. The LLM Experiment

This is where the "humans and LLMs" theme comes alive. Include in your report:

1. Write **10 small tasks** (gcd, fibonacci, is_prime, ...; 2 should use function values)
2. Give an LLM only your `SPEC.md` and one task at a time
3. Compile each output. Record: compiles? correct? error category?
4. **Repair loop:** feed your compiler's error message back once. How many failures does the message fix?
5. Based on what you saw, change one thing in your spec, grammar, or error messages. Re-measure.

Your grade does **not** depend on a high success rate. It depends on honest measurement and a specific conclusion: what makes a language LLM-friendly?

Also compare BPE token counts (`tiktoken`) of your example programs vs equivalent Python/OCaml.

---

## 8. Deliverables

```
final-project/
├── README.md          group, path, LLVM version, build/run instructions
├── SPEC.md            grammar, tokens, precedence, type rules, examples
├── src/               lexer, parser, AST, type checker, code generator
├── runtime.c          print built-ins
├── tests/             valid/ errors/ recovery/
├── examples/          .ll output for 2-3 demo programs
├── run_tests.sh       one-command test runner
└── REPORT.pdf         3 pages
```

### REPORT.pdf (3 pages)

1. **Language design**: goals, three design decisions with alternatives rejected, how it serves humans and LLMs
2. **Architecture and testing**: pipeline diagram, how recovery works, three real bugs you found, test counts
3. **LLM experiment**: results, one concrete change you made based on findings, and what you learned

Plus: AI usage disclosure (tools used, 2-3 prompts verbatim, one AI mistake you caught). If no AI: "No AI tools were used."

### Demo (10 minutes, Week 15)

- One slide: your language
- Live demo: valid program → output, program with errors → diagnostics, AST/IR dump
- The hardest bug
- LLM experiment results
- Each member explains one component

I may introduce a small mutation (e.g., change `add` to `sub` in your code generator). Your test suite should catch it.

---

## 9. Grading

| Component | Points | What's graded |
|-----------|--------|---------------|
| **Phase 1** | 30 | Clear spec, correct parser, real AST, located errors, recovery, tests |
| **Phase 2** | 30 | Type rules correct, scoping/shadowing, good diagnostics, multi-error recovery, tests |
| **Phase 3** | 40 | Valid IR that runs, all features work, division-by-zero, no crashes, full test suite |

**Report and demo** are folded into the phase grades (especially Phase 3). Bonus work (Section 10) is extra.

Partial credit is given for incomplete phases — submit what works and document what's missing.

---

## 10. Bonus (+10%, pick any)

- **Closure conversion**: support capturing lambdas with closure structs
- **Hindley–Milner inference**: drop annotations, support polymorphic `let`
- **Lists or algebraic data types** with pattern matching
- **Tail-call elimination** (`musttail` or loop lowering)
- **Machine-readable diagnostics** (`--diagnostics=json`)
- Or propose your own

---

## FAQ

**Must we write everything from scratch?**
You write the AST, type checker, and code generator. Flex, Bison, PLY, Menhir, and LLVM are tools — use them.

**Can we interpret instead of generating LLVM?**
An interpreter is a great testing oracle (and I recommend it), but LLVM output is required.

**Do we need strings or floats?**
No. `int` and `bool` are sufficient.

**What if we don't finish a feature?**
Remove it from the language or reject it with a diagnostic. Document it as unsupported.

**Do we need a GPU?**
No.

**How big is this project?**
Roughly: ~1000 lines of OCaml/Python, or a few thousand lines of C. The tests and diagnostics are where the rest of the time goes.

---

## Timeline

| Week | Target |
|------|--------|
| 8 | Form teams, choose path, design language. Get a trivial program (`print_int 42`) through every stage. |
| 9–10 | Phase 1: parser, AST, syntax errors, recovery |
| 11–12 | Phase 2: type checker, diagnostics, recovery |
| 13–14 | Phase 3: LLVM codegen, full test suite, LLM experiment, report |
| 15 | Demos |

---

## Resources

- Course examples: `lecture-notes/5-bison-example/`, `lecture-notes/8-semantic-analysis-type-checking/`
- LLVM Kaleidoscope tutorial: https://llvm.org/docs/tutorial/
- Crafting Interpreters: https://craftinginterpreters.com
- LLVM Language Reference: https://llvm.org/docs/LangRef.html
- tiktoken: https://github.com/openai/tiktoken

---

*Get a trivial program through every stage in the first week. Then grow the language.*