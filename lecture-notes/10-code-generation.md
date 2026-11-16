## **Code Generation: From the Typed AST to Executable Code**

BIL 463 Compiler Design · Week 10

### **The Final Compiler Phase**

```
Front end (done)                Back end (today)
AST + symbol table     ->      LLVM IR (your project)  or  machine code (MIPS, by hand)
Type information       ->      Executable program
Semantic checks        ->      Run-time efficiency
```

### **Key Challenge**

**How do we translate our understanding of the program (a typed AST) into instructions that run?**

By the end of this lecture you can

- explain intermediate representations and **SSA**
- lower each construct of your language (`let`, `if`, `&&`, calls, function values, division) to **LLVM IR**, and check it with the verifier
- explain how a classic code generator (stack machine, MIPS) works, and compare it with the LLVM route
- test a code generator with **differential testing** (reference interpreter, `-O0` vs `-O2`)
- place code generation in the AI era: ML compilers, LLMs as code writers, verification

---

## **Intermediate Representations**

### **Why Not Generate Machine Code Directly?**

An IR sits between the front end and the machine: n languages and m machines need n + m compilers instead of n x m, and optimizations are written once.

- **Graphical IRs**
  - syntax trees: parse tree, AST
  - graphs: control-flow graph (CFG), data-dependence graph, call graph
- **Linear IRs**
  - stack machine code
  - one-, two-, three-address code (`t5 = t4 - t3`)
  - SSA form (LLVM IR, GCC GIMPLE)

### **One Expression, Four Forms**

Source: `let x = y + 5 in x * 2`

```
AST:                    Three-address code:     LLVM IR (SSA):             MIPS (stack machine):
let x                   t1 = y + 5              %x = add i32 %y, 5         lw  $a0, 4($fp)
 |- (+ y 5)             x  = t1                 %r = mul i32 %x, 2         ... push, add ...
 '- (* x 2)             t2 = x * 2                                         ... mul
                                                ret i32 %r
```

Notice what **disappeared** in LLVM IR: `let x` costs no instruction. `x` is just a name for the value `%x`. Immutable bindings are exactly what SSA wants.

---

## **Code Generation Strategies**

### **Three Styles**

**1. Stack machine** (interpreters, virtual machines)

```
push 5
push 3
add
store result
```

(Real bytecode looks like this: JVM `iconst_5, iconst_3, iadd, istore_1`.)

**2. Register machine** (traditional compilers, real CPUs)

```asm
# x86-64 (Intel syntax)
mov rax, 5
add rax, 3
mov [result], rax
```

**3. SSA IR** (LLVM, GCC)

```llvm
%1 = add i32 5, 3
store i32 %1, ptr @result
```

> SSA (**Static Single Assignment**) is a property of an IR, not a third kind of machine: every name is assigned **exactly once**, and `phi` nodes merge values where control flow joins. An SSA IR is later lowered to a register machine. Since LLVM 15 opaque pointers are the default: a pointer is just `ptr`, not `i32*` (typed pointers were removed in later releases).

### **Why Start with Stack Machines?**

- **Simple** to understand and implement
- **Uniform** code generation pattern: one template per AST node
- A **foundation** for the other styles (and WebAssembly is a stack machine)

---

## **Stack Machine Fundamentals**

### **The Simple Evaluation Model**

- `push value`: push a value
- `pop`: remove the top value
- `add`: pop two values, push their sum
- `sub`, `multiply`: same pattern

**Example `(7 + 5) * 2`**

```
push 7      [7]
push 5      [7, 5]
add         [12]
push 2      [12, 2]
multiply    [24]
```

### **Real-World Stack Machines**

- Java bytecode (JVM)
- WebAssembly
- Python bytecode
- .NET CIL

Many fast VMs are register based instead (Lua 5, Android's Dalvik), but a stack machine is the easiest target to generate.

---

# **Part 1: Generating LLVM IR (Your Final Project)**

---

## **The Final Project Target: Textual LLVM IR**

### **The Pipeline of Phase 3**

```
prog.fpl --lexer/parser--> AST --type checker--> typed AST --code generator--> prog.ll
prog.ll  --opt (optional)--> prog.ll --llc--> assembly --assembler/linker--> executable
                                         (clang drives these steps; runtime.c is linked in)
```

```bash
./fplc prog.fpl --emit-ll -o prog.ll         # your compiler
opt -passes=verify -disable-output prog.ll   # is the IR well formed?
clang prog.ll runtime.c -o prog              # LLVM compiles the IR, links the C runtime
./prog
```

- **Emit readable `.ll` text yourself.** You see exactly what you produce.
- Use LLVM 15 or newer (opaque `ptr`) and say in your README which version you tested.
- LLVM then does what this lecture's second part does by hand: instruction selection, register allocation, frames, for x86-64, ARM64, RISC-V, and more.

---

## **Anatomy of an LLVM IR File**

```llvm
declare i32 @print_int(i32)                 ; provided by runtime.c

define i32 @double(i32 %x) {                ; function: return type, @name, typed parameters
entry:                                      ; a basic block: a label ...
  %t = mul i32 %x, 2                        ; ... instructions: %name = opcode type operands
  ret i32 %t                                ; ... and exactly ONE terminator (ret, br)
}

define i32 @main() {
entry:
  %r = call i32 @double(i32 21)
  %p = call i32 @print_int(i32 %r)
  ret i32 0
}
```

- `@name` is global (functions), `%name` is local (values, parameters). Comments start with `;`.
- **Types are explicit** on every operand: `i32` (integers), `i1` (booleans), `ptr` (any pointer, including function pointers), `void`.
- Every **basic block** ends with a **terminator**: `ret`, `br`, `unreachable`.
- Values are **SSA**: each `%name` is defined once and never changed.
- Functions may be called **before** they are defined in the file, so mutual recursion needs no forward declarations.
- Prefer **named** temporaries (`%t.1`). Unnamed ones (`%0`, `%1`) must be numbered in order, which is easy to get wrong.

---

## **From Your Language to LLVM IR**

### **The Mapping**

| Your language | LLVM IR |
|---------------|---------|
| `int` (32-bit, wraps) | `i32`, with plain `add`, `sub`, `mul` (no `nsw` flag) |
| `bool` | `i1` (`true`, `false`); `zext i1 %b to i32` when the C runtime wants an int |
| function type `int -> int` | `ptr` |
| integer literal, `true` | constants: `5`, `true` |
| name bound by `let` or a parameter | an SSA **value**: `%x` |
| `e1 + e2`, `-`, `*` | `add`, `sub`, `mul` |
| unary `-e` | `sub i32 0, %e` |
| `e1 < e2` and the other comparisons | `icmp slt` (`sle`, `sgt`, `sge`, `eq`, `ne`), result `i1` |
| `e1 / e2` | zero check, then `sdiv` |
| `let x = e1 in e2` | **no instruction**: generate `e1`, bind `x` to its value |
| `if c then a else b` | `br i1`, two blocks, a join block with `phi` |
| `a && b`, `a || b` | `br` for short-circuit, join with `phi` |
| call `f a b` | `call i32 @f(i32 %a, i32 %b)` |
| call of a function value `f x` | `call i32 %f(i32 %x)` |
| anonymous function `fun (x: int) -> e` | a **new top-level function**, value `ptr @lambda.1` |

### **The Core Idea**

```
gen(e, env)   emits instructions for e and RETURNS the operand holding its value
env           maps each name in scope to an LLVM operand   (week 8: names -> types; now: names -> values)
```

- The type checker already decided every type, so the generator never checks anything: it **trusts the typed AST**.
- Generating an expression returns a **constant** (`5`) or a **name** (`%t.4`). The caller uses it as an operand.
- The generator also remembers **which basic block it is currently in**. This matters for `phi`.

---

## **Expressions, Variables, Let**

### **Arithmetic and Comparison**

```
gen(Int n)         = "n"                                  (no instruction)
gen(Bin(+, a, b))  = va = gen(a); vb = gen(b)
                     emit  %t = add i32 va, vb ;  return %t
gen(Bin(<, a, b))  = ...  emit  %t = icmp slt i32 va, vb ;  return %t      (an i1)
```

```llvm
; 2 * 3 + 4
%t.1 = mul i32 2, 3
%t.2 = add i32 %t.1, 4
```

Two semantic decisions your spec fixes, and the IR must follow:

- **Wrap-around:** `add i32` wraps. The flag `nsw` ("no signed wrap") would make overflow *poison* (undefined behavior): do not add it. `2147483647 + 1` must print `-2147483648`.
- **Evaluation order:** left to right, strict. `gen(a)` before `gen(b)` matters because `print_int` has side effects.

### **let: Just a Name**

```
gen(Let(x, e1, e2), env) = v = gen(e1, env)
                           return gen(e2, env + {x -> v})
```

```
let x = 5 in
let y = x * 2 in
y + x
```

```llvm
%t.1 = mul i32 5, 2      ; y   (x is just the constant 5: it cost no instruction)
%t.2 = add i32 %t.1, 5
```

- **No stack slot, no `alloca`.** Immutable bindings make this possible: functional languages are easy to compile to SSA.
- **Shadowing needs no work:** an inner `x` replaces the outer one in `env`; outside the `let` the old `env` is still in use.
- (A language with mutable variables would use `alloca`, `load`, `store`, and let LLVM's `mem2reg` pass rebuild SSA. Clang does this for C.)

---

## **If: Branches and phi**

### **The Rule**

```
gen(If(c, a, b)) =
    vc = gen(c)
    emit  br i1 vc, label %then, label %else
    then:  va = gen(a) ; then_end = (current block) ; emit br label %join
    else:  vb = gen(b) ; else_end = (current block) ; emit br label %join
    join:  emit  %t = phi T [ va, %then_end ], [ vb, %else_end ] ; return %t
```

```llvm
; if n <= 1 then 1 else n * fact (n - 1)
  %c = icmp sle i32 %n, 1
  br i1 %c, label %then, label %else
then:
  br label %join
else:
  %n1 = sub i32 %n, 1
  %r  = call i32 @fact(i32 %n1)
  %m  = mul i32 %n, %r
  br label %join
join:
  %v = phi i32 [ 1, %then ], [ %m, %else ]
  ret i32 %v
```

`phi` says: "the value is `1` if we came from `%then`, `%m` if we came from `%else`". It must be the first instruction of the join block, with one entry for every predecessor.

### **The Classic Bug: Which Block Is the Predecessor?**

```
if a then (if b then 1 else 2) else 3
```

The `then` branch **contains another `if`**. By the time it finishes, the current block is the inner join (`join2`), not the block you started with (`then`). The `phi` of the outer `if` must name the block you are **in when you emit the branch to the join**: `then_end`, not `then`.

```llvm
then:
  br i1 %b, label %t2, label %e2        ; the inner if starts in %then ...
t2:
  br label %join2
e2:
  br label %join2
join2:                                  ; ... and ends in %join2
  %x = phi i32 [ 1, %t2 ], [ 2, %e2 ]
  br label %join                        ; so the predecessor of %join on this path is %join2
else:
  br label %join
join:
  %y = phi i32 [ %x, %join2 ], [ 3, %else ]     ; correct. Writing [ %x, %then ] is the bug
```

With `%then` the verifier answers (real message):

```
PHI node entries do not match predecessors!
  %y = phi i32 [ %x, %then ], [ 3, %else ]
label %then
label %else
Instruction does not dominate all uses!
...
```

---

## **Short-Circuit && and ||**

### **The Right Operand May Not Run**

`false && say 1` must **not** call `say`: it prints. So `&&` is control flow, like `if`:

```
a && b   =   if a then b else false
a || b   =   if a then true else b
```

```llvm
; a && b       (a is %a, b is generated into the block %rhs)
entry:
  br i1 %a, label %rhs, label %join
rhs:
  ; ... code for b, result %b ...
  br label %join
join:
  %t = phi i1 [ false, %entry ], [ %b, %rhs ]
```

- `||` swaps the branch targets and the constant (`true`).
- Again the `phi` entries are the blocks in which the branches end (`entry`, and the **last block of `b`**, which may not be `%rhs` if `b` contains control flow).
- Do **not** compile `&&` as `and i1 %a, %b`: that evaluates both sides.

---

## **Functions, Calls, Function Values**

### **Definitions and Calls**

```
let fact (n: int): int = ...        define i32 @fact(i32 %n) { entry: ... ret i32 %v }
fact (n - 1)                        %r = call i32 @fact(i32 %n1)
let both (a: bool) (b: bool): bool  define i1 @both(i1 %a, i1 %b) { ... }
```

- **Signatures first (two passes):** collect every top-level function's type, then generate bodies. This is the same two-pass idea as the type checker, and it makes **mutual recursion** work.
- Parameters are values: `env = { n -> %n }`.
- Built-ins: `declare i32 @print_int(i32)`; for `print_bool`, convert `i1` to `i32` with `zext` before the call.

### **Function Values and Lambdas**

Your functions may be **passed as arguments** and **called**. A function name used as a value is a `ptr`:

```
let apply (f: int -> int) (x: int): int = f x
apply double 5
```

```llvm
define i32 @apply(ptr %f, i32 %x) {
entry:
  %t = call i32 %f(i32 %x)                 ; indirect call through a ptr
  ret i32 %t
}
; ... in main:
  %r = call i32 @apply(ptr @double, i32 5)
```

An anonymous function that **captures nothing** is **lifted** to a top-level function with a fresh name, and its value is a pointer to it:

```
apply (fun (y: int) -> y + 1) 5
```

```llvm
define i32 @lambda.1(i32 %y) {
entry:
  %t.1 = add i32 %y, 1
  ret i32 %t.1
}
; ...
  %t.2 = call i32 @apply(ptr @lambda.1, i32 %t.1)
```

Why the no-capture restriction? A lambda that uses a local variable needs that variable at run time: a **closure** (last week): a code pointer **plus** an environment record on the heap. Your type checker rejects it; the closure bonus implements it.

---

## **Division by Zero**

### **Why a Check?**

In LLVM, `sdiv` by zero is **undefined behavior**: the program may crash, return garbage, or be optimized in surprising ways. Your spec says: stop with a message on stderr and a **nonzero exit code**.

```llvm
%z = icmp eq i32 %b, 0
br i1 %z, label %divzero, label %divok
divzero:
  call void @fpl_div_zero()              ; runtime.c: prints to stderr, exit(1)
  unreachable                            ; this call never returns
divok:
  %q = sdiv i32 %a, %b
```

```c
/* runtime.c */
#include <stdio.h>
#include <stdlib.h>
int print_int(int x)  { printf("%d\n", x); return 0; }
int print_bool(int b) { printf("%s\n", b ? "true" : "false"); return 0; }
void fpl_div_zero(void) { fprintf(stderr, "runtime error: division by zero\n"); exit(1); }
```

- `unreachable` tells LLVM control never gets there; it is also the terminator the block needs.
- **Edge case beyond the spec:** `-2147483648 / -1` overflows and is also undefined for `sdiv` (the x86 CPU even traps). Decide in your SPEC whether you check it too.

---

## **Programs, main, and the Runtime**

### **The Shape of a Program**

A program is top-level definitions followed by **one final expression**. The final expression is the body of `@main`; its value is discarded and `main` returns 0.

```
let double (x: int): int = x * 2
let apply (f: int -> int) (x: int): int = f x
let fact (n: int): int = if n <= 1 then 1 else n * fact (n - 1)
let answer =
  let x = 5 in
  let y = apply double x in
  if y > fact 3 then y else fact 3

print_int answer
```

- Top-level **functions** become `define`s.
- Top-level **values** (`let answer = ...`) are evaluated **in order, before the final expression**. Easiest design: generate them inside `@main` as SSA values. Decide (and document) whether functions may refer to them; if yes you need global variables.
- `@main` ends with `ret i32 0`.

### **The Generated IR (excerpt, produced by the lecture's mini compiler)**

```llvm
define i32 @main() {
entry:
  %t.1 = call i32 @apply(ptr @double, i32 5)
  %t.2 = call i32 @fact(i32 3)
  %t.3 = icmp sgt i32 %t.1, %t.2
  br i1 %t.3, label %then.4, label %else.5
then.4:
  br label %join.6
else.5:
  %t.7 = call i32 @fact(i32 3)
  br label %join.6
join.6:
  %t.8 = phi i32 [ %t.1, %then.4 ], [ %t.7, %else.5 ]
  %t.9 = call i32 @print_int(i32 %t.8)
  ret i32 0
}
```

It prints `10`. (`let x` and `let y` disappeared into the operands, as promised.)

---

## **The Verifier Is Your Best Friend**

### **Check Every Emitted File**

```bash
opt -passes=verify -disable-output prog.ll       # silent when the IR is well formed
```

The project requires **every `.ll` to pass the verifier**. Typical messages, with the mistake behind them:

```
; use a value from a branch that does not dominate the use
define i32 @f(i1 %c) {
entry:  br i1 %c, label %then, label %join
then:   %x = add i32 1, 2
        br label %join
join:   ret i32 %x                    ; x only exists on one path
}
-> Instruction does not dominate all uses!
```

```
%r = add i32 %c, 1                    ; %c is an i1 parameter
-> '%c' defined with type 'i1' but expected 'i32'
```

```
entry:
  %r = add i32 %a, 1                  ; no ret / br: the block has no terminator
}
-> expected instruction opcode
```

| Symptom | Usual cause |
|---------|-------------|
| `PHI node entries do not match predecessors` | wrong block names in `phi` (nested `if` / `&&`), or a branch missing |
| `Instruction does not dominate all uses` | using a value defined inside a branch after the join |
| `defined with type i1 but expected i32` | `bool` and `int` confused; missing `zext` |
| `expected instruction opcode` or unterminated block | a block without `ret` or `br` |
| duplicate label or name | counters not unique (use one counter per function) |

### **Reading and Experimenting**

- `clang -S -emit-llvm -O0 prog.c`: see the IR clang produces for equivalent C (`alloca` and `load`/`store` everywhere); with `-O1` it is SSA. A great way to **learn IR idioms**.
- `lli prog.ll`: interpret or JIT an `.ll` file; `llvm-dis` and `llvm-as` convert between text and bitcode.
- Language Reference: https://llvm.org/docs/LangRef.html

---

## **After Your IR: LLVM Does the Rest**

### **Optimizations (opt) and Code Generation (llc)**

Running `opt -passes='default<O2>'` on the IR of the example above gives (real output, trimmed):

```llvm
define i32 @fact(i32 %n) {
entry:
  %t.11 = icmp slt i32 %n, 2
  br i1 %t.11, label %join.4, label %else.3
else.3:                                           ; preds = %entry, %else.3
  %n.tr3 = phi i32 [ %t.5, %else.3 ], [ %n, %entry ]
  %accumulator.tr2 = phi i32 [ %t.7, %else.3 ], [ 1, %entry ]
  %t.5 = add nsw i32 %n.tr3, -1
  %t.7 = mul i32 %n.tr3, %accumulator.tr2
  ...
define i32 @main() {
  %t.9 = tail call i32 @print_int(i32 10)
  ret i32 0
}
```

- **Constant folding and inlining:** `main` computes `10` at compile time.
- **Tail recursion elimination:** the recursion in `fact` became a loop (`phi` of an accumulator).
- Other classic passes: dead code elimination, common subexpression elimination, `mem2reg`, loop optimizations.

### **Back End: Instruction Selection and Register Allocation**

The same IR for `calculate(x, y) = x * y + 10` through `llc -O2` for different targets (real output):

```asm
# x86-64                    # AArch64                      # RISC-V (rv64im)         # MIPS32
calculate:                  calculate:                     calculate:                 calculate:
  imull %esi, %edi            mov  w8, #10                   mul   a0, a0, a1           mul   $1, $4, $5
  leal  10(%rdi), %eax        madd w0, w0, w1, w8            addiw a0, a0, 10           jr    $ra
  retq                        ret                            ret                        addiu $2, $1, 10
```

- **Instruction selection** picks instructions (`lea` and `madd` fuse operations), **register allocation** decides which values live in which registers, and the **calling convention** (last lecture) decides that arguments arrive in `edi/esi`, `w0/w1`, `a0/a1`, `$4/$5`.
- The MIPS version has a **branch delay slot**: the instruction after `jr` still executes.

> You emit IR; LLVM provides optimization, register allocation, and four ISAs. That is the point of a good IR.

---

# **Part 2: Generating Code by Hand: Stack Machine and MIPS**

---

## **Why Also Learn the Classic Way?**

- It shows **what the IR hides**: stack discipline, frames, calling sequence, register use. These are exam topics and are what LLVM does for you.
- The **same recursive pattern** (one template per AST node) appears in every code generator, including the LLVM one above.
- MIPS is a clean RISC target that you can read and simulate (SPIM, MARS). (Other RISC targets, like RISC-V, look very similar.)
- Reference implementation for this part: `mips_cgen.py` in the example folder (tested in a small simulator and assembled with LLVM's MIPS assembler).

---

## **Introduction to MIPS**

### **Why MIPS for Teaching Code Generation?**

**MIPS** (Microprocessor without Interlocked Pipeline Stages) is a RISC architecture born at Stanford and the standard teaching ISA of the textbooks.

- **Regular instruction set**, fixed 32-bit instructions
- **Load-store model:** only `lw`/`sw` touch memory, ALU instructions work on registers
- **Simple addressing**
- Good simulators (SPIM, MARS); the company behind it now designs RISC-V cores, whose assembly looks very similar

### **MIPS vs x86-64**

| Aspect | **MIPS** | **x86-64** |
|--------|----------|------------|
| **Architecture** | RISC | CISC |
| **Registers** | 32 general-purpose (`$0` to `$31`) | 16 general-purpose (`rax`, `rbx`, ...) |
| **Instruction format** | Fixed 32-bit, regular | Variable length, complex |
| **Memory access** | Load/store only | Memory operands in most instructions |
| **Calling convention** | `$a0` to `$a3` arguments, `$v0` and `$v1` results | `rdi, rsi, ...` (System V) or `rcx, rdx, ...` (Windows) |
| **Teaching value** | Simple, predictable | Historical quirks, but what most PCs run |

### **MIPS Registers You Need**

```mips
$a0-$a3     # argument registers
$v0-$v1     # return values (and syscall numbers)
$t0-$t9     # temporaries (caller-saved)
$s0-$s7     # saved registers (callee-saved)
$gp $sp $fp # global pointer, stack pointer, frame pointer
$ra         # return address
```

Reference card: https://www.cs.tufts.edu/comp/140/lectures/Day_3/mips_summary.pdf

### **MIPS Instruction Examples**

```mips
# Load / store
lw   $t0, 4($sp)        # load word from memory (sp + 4) into $t0
sw   $a0, 0($sp)        # store word from $a0 to memory (sp)

# Arithmetic
add  $a0, $t1, $t2      # $a0 = $t1 + $t2
sub  $a0, $t1, $t2
mul  $a0, $t1, $t2
slt  $a0, $t1, $t2      # $a0 = ($t1 < $t2) ? 1 : 0

# Control flow
beq  $t0, $t1, label    # branch if equal
j    label              # jump
jal  func               # jump and link (call): $ra = return address
jr   $ra                # jump to return address

# Immediates and pseudo-instructions
addiu $t0, $t1, 5       # $t0 = $t1 + 5
li   $t0, 42            # load immediate: PSEUDO-instruction
move $t0, $t1           # PSEUDO: addu $t0, $zero, $t1
b    label              # PSEUDO: beq $zero, $zero, label
```

`li`, `move`, `b`, and many more are **pseudo-instructions**: the assembler expands them into real ones. Real MIPS also has a **branch delay slot** (the instruction after a branch executes too); SPIM and MARS let you turn delay slots off, and the examples here assume they are off.

---

## **Expressions on a Stack Machine, in MIPS**

### **Example: `a + b * c`**

```
      +
     / \
    a   *
       / \
      b   c
```

With a register allocator (what a good compiler does):

```mips
# a in $s0, b in $s1, c in $s2, result in $a0
mul  $t0, $s1, $s2      # b * c
add  $a0, $s0, $t0      # a + (b * c)
```

The **stack machine approach** (what a simple code generator does, one template per node):

```
push a      # [a]
push b      # [a, b]
push c      # [a, b, c]
multiply    # [a, (b*c)]
add         # [a+(b*c)]
```

### **Simulating the Stack on MIPS: the Accumulator Convention**

- **Accumulator** `$a0`: the value of the last expression evaluated
- **Stack pointer** `$sp`: points to the **next free** word (the stack grows down)
- **Temporary** `$t1`: for the left operand
- **Frame pointer** `$fp`: a stable base for locals and arguments

```mips
# push the accumulator
sw    $a0, 0($sp)
addiu $sp, $sp, -4

# pop into $t1
lw    $t1, 4($sp)
addiu $sp, $sp, 4
```

---

## **The Code Generator: One Template per Node**

### **Recursive AST Traversal**

The invariant: **after `cgen(e)` the value of `e` is in `$a0` and the stack is as it was before.**

```
cgen(Int n)        = li   $a0, n
cgen(Var x)        = lw   $a0, offset(x)($fp)           # names -> frame offsets
cgen(a + b)        = cgen(a)
                     sw   $a0, 0($sp) ; addiu $sp, $sp, -4     # push left
                     cgen(b)
                     lw   $t1, 4($sp)  ; addiu $sp, $sp, 4     # pop left into $t1
                     add  $a0, $t1, $a0
```

For `-` the order matters (`$t1` is the **left** operand): `sub $a0, $t1, $a0`.

```python
def cgen(node):                         # sketch: emit a list of instructions, do not build strings
    if node.kind == "int":
        emit(f"li   $a0, {node.value}")
    elif node.kind == "binop":
        cgen(node.left)
        push()                          # sw $a0, 0($sp) ; addiu $sp, $sp, -4
        cgen(node.right)
        pop("$t1")                      # lw $t1, 4($sp) ; addiu $sp, $sp, 4
        emit({"+": "add", "-": "sub", "*": "mul"}[node.op] + "  $a0, $t1, $a0")
    else:
        raise NotImplementedError(node.kind)     # never return None silently
```

### **Example: `3 + (7 + 5)`**

```mips
li   $a0, 3
sw   $a0, 0($sp)           # push 3
addiu $sp, $sp, -4
li   $a0, 7
sw   $a0, 0($sp)           # push 7
addiu $sp, $sp, -4
li   $a0, 5
lw   $t1, 4($sp)           # pop 7
addiu $sp, $sp, 4
add  $a0, $t1, $a0         # 7 + 5 = 12
lw   $t1, 4($sp)           # pop 3
addiu $sp, $sp, 4
add  $a0, $t1, $a0         # 3 + 12 = 15
```

---

## **let, if and Variables: Names to Frame Offsets**

### **let in the Stack Machine**

```
cgen(Let(x, e1, e2)) = cgen(e1)                      # value in $a0
                       push                          # x lives in the slot just pushed
                       cgen(e2) with env + { x -> that slot }
                       addiu $sp, $sp, 4             # drop x; the result stays in $a0
```

The environment now maps names to **offsets from `$fp`** instead of to types. The **same scoping rules** as week 8 decide which `x` is visible; the stack discipline decides where it lives. A `let` variable pushed when `d` words are already pushed sits at `-4*(d+1)($fp)`; parameters sit at **positive** offsets.

### **if**

```mips
        cgen(cond)
        beq  $a0, $zero, else_1     # 0 is false
        cgen(then_expr)
        b    endif_2
else_1: cgen(else_expr)
endif_2:                            # the result is in $a0 on both paths
```

Labels must be **unique** (use a counter). Booleans are 0 and 1. `&&` and `||` are the same short-circuit idea: skip the right operand with `beq` or `bne`.

### **Comparisons**

```mips
slt  $a0, $t1, $a0          # e1 < e2
slt  $a0, $a0, $t1          # e1 > e2   (swap the operands)
slt  $a0, $a0, $t1 ; xori $a0, $a0, 1      # e1 <= e2  =  not (e2 < e1)
xor  $a0, $t1, $a0 ; sltiu $a0, $a0, 1     # e1 == e2
```

(`sgt`, `seq` and friends are pseudo-instructions and expand to similar sequences.)

---

## **Function Calls and Activation Records**

### **The Calling Convention Used Here (Cool / Stanford CS143)**

**Caller:**
1. push the caller's frame pointer
2. evaluate and push the arguments, **last argument first**
3. `jal f`

**Callee:**
1. `$fp = $sp`, then push the return address `$ra`
2. run the body (the result is left in `$a0`)
3. reload `$ra`, **pop** the return address, the arguments and the saved frame pointer
4. restore `$fp`, `jr $ra`

```mips
# caller: f(x, y)   (x pushed last, so it ends up nearest the callee)
sw   $fp, 0($sp)       # save the frame pointer
addiu $sp, $sp, -4
cgen(y) ; push         # last argument first
cgen(x) ; push
jal  f

# callee: f has n = 2 arguments
f:
  move  $fp, $sp                 # $fp points at the (still empty) slot for $ra
  sw    $ra, 0($sp)
  addiu $sp, $sp, -4
  # ... body, result in $a0 ...
  lw    $ra, 4($sp)
  addiu $sp, $sp, 16             # pop $ra, x, y, and the saved $fp slot: 4*n + 8
  lw    $fp, 0($sp)              # restore the caller's frame pointer
  jr    $ra
```

### **The Frame, Drawn Correctly**

For `f(x, y)` (the stack grows **down**; arguments are **above** the frame pointer):

```
High address
┌───────────────┐
│ saved $fp     │  $fp + 12   (pushed by the caller)
├───────────────┤
│ y             │  $fp + 8
├───────────────┤
│ x             │  $fp + 4
├───────────────┤
│ return addr   │  <- $fp points here (0($fp)), pushed by the callee
├───────────────┤
│ let x, temps  │  $fp - 4, $fp - 8, ...
│ ...           │
└───────────────┘  <- $sp (next free word)
Low address
```

**Variable access:**

```mips
lw  $a0, 4($fp)      # parameter x
lw  $a0, 8($fp)      # parameter y
lw  $a0, -4($fp)     # first let-bound variable or temporary
```

Compare with last week's x86-64 frame: same ingredients (return address, saved frame pointer, arguments, locals); a different **convention** about who pushes what.

---

## **Worked Example: A Function with let**

```
let g (y: int): int = let x = y + 5 in x * 2
```

```mips
g:
    move $fp, $sp
    sw   $ra, 0($sp)
    addiu $sp, $sp, -4
    lw   $a0, 4($fp)          # y
    sw   $a0, 0($sp)
    addiu $sp, $sp, -4        # push y
    li   $a0, 5
    lw   $t1, 4($sp)
    addiu $sp, $sp, 4         # pop y
    add  $a0, $t1, $a0        # y + 5
    sw   $a0, 0($sp)
    addiu $sp, $sp, -4        # x lives here: -4($fp)
    lw   $a0, -4($fp)         # x
    sw   $a0, 0($sp)
    addiu $sp, $sp, -4
    li   $a0, 2
    lw   $t1, 4($sp)
    addiu $sp, $sp, 4
    mul  $a0, $t1, $a0        # x * 2
    addiu $sp, $sp, 4         # drop x
    lw   $ra, 4($sp)
    addiu $sp, $sp, 12        # 4*1 + 8
    lw   $fp, 0($sp)
    jr   $ra
```

Compare with the LLVM IR for the same function, which is **two instructions** and no memory traffic:

```llvm
define i32 @g(i32 %y) {
entry:
  %x = add i32 %y, 5
  %r = mul i32 %x, 2
  ret i32 %r
}
```

The stack-machine version is simple and uniform but slow; a **register allocator** (in LLVM's back end) removes the memory traffic.

---

## **Object-Oriented Code Generation**

### **Objects and Virtual Method Tables**

```java
class Point {
    int x, y;
    void move(int dx, int dy) { x = x + dx; y = y + dy; }
}
```

An object in memory (Cool-style layout):

```
Point object:
┌─────────────┐
│ Class tag   │   runtime type information
├─────────────┤
│ Object size │   for the garbage collector
├─────────────┤
│ Dispatch ptr│   -> the class's virtual method table
├─────────────┤
│ x           │   instance variables at fixed offsets
├─────────────┤
│ y           │
└─────────────┘
```

Type information from semantic analysis gives the field **offsets**: `p.x` becomes a load at a constant offset from the object pointer.

### **Dynamic Dispatch**

```
Animal VTable    Dog VTable      Cat VTable
┌─────────────┐  ┌─────────────┐ ┌─────────────┐
│  speak()    │  │  speak()    │ │  speak()    │
│  eat()      │  │  eat()      │ │  eat()      │
│  sleep()    │  │  sleep()    │ │  sleep()    │
└─────────────┘  │  bark()     │ │  meow()     │
                 └─────────────┘ └─────────────┘
```

A subclass keeps the **same slot numbers** for inherited methods; overriding replaces the address in the slot.

```mips
# animal.speak()    (the object pointer is in $a0)
lw   $t0, 8($a0)     # load the dispatch pointer from the object
lw   $t1, 0($t0)     # load the address of speak() (slot 0)
jalr $t1             # call it, with the receiver as the first argument
```

Variations you will meet: Java **interface** dispatch tables, C++ multiple inheritance, Rust trait objects (a **fat pointer**: data pointer plus vtable pointer), and JIT **inline caches** that skip the lookup when the receiver type is stable.

### **The Functional Analogue**

- A **closure** is an object with one method: code pointer plus captured values.
- A **variant** (`type shape = Circle of int | Rect of int * int`) is an object with a **tag** and fields; `match` compiles to a **switch on the tag**, exactly like dispatch on the class tag above. (Bonus topic for your project.)

---

## **Modern Code Generation Trends**

### **Just-In-Time (JIT) Compilation**

- **Tiered compilation:** start in an interpreter, compile hot code
- **Profile-guided optimization:** optimize for what actually runs
- **Deoptimization:** if an assumption fails, fall back to the interpreter

### **WebAssembly: A Portable Stack Machine**

```wat
;; WebAssembly text format
(func $add (param $x i32) (param $y i32) (result i32)
  local.get $x
  local.get $y
  i32.add)
```

The stack-machine style from the start of this lecture, now a real target for browsers and servers. Wasm 3.0 (2025) added garbage collection support and tail calls, which makes it a practical target for functional languages. (Generating Wasm from your typed AST is a nice extension of the project.)

### **Machine Learning Compilers**

Deep learning frameworks are compilers too: a model is a **graph of tensor operations** (the IR), optimized and lowered to GPU or accelerator code.

| Tool | What it is |
|------|------------|
| **MLIR** | reusable compiler infrastructure with multiple levels of IR ("dialects"), part of the LLVM project |
| **XLA** | open-source compiler for machine learning (used by JAX and TensorFlow); fuses operations and plans buffers |
| **Apache TVM** | an open machine learning compiler framework |
| **Triton** | a language and compiler for writing custom GPU kernels |
| **`torch.compile`** | PyTorch 2's compiler; its default backend, TorchInductor, uses Triton for GPUs |

This is the same pipeline as this course: front end (capture the model), typed IR (tensor shapes, week 8), optimization (fusion, memory planning), and code generation for a target that is a GPU instead of a CPU.

---

## **Code Generation and AI**

### **LLMs as Code Writers and as Compiler Researchers**

- **LLMs write assembly and IR badly, and compile errors are cheap feedback.** The verifier, the assembler and the test suite are the judge. The loop from week 8 (generate, verify, repair) applies directly.
- **Research directions:** Meta's *Large Language Model Compiler* is a model trained on LLVM IR and assembly (546 billion tokens) for tasks such as code-size optimization and disassembly. DeepMind's AlphaDev used reinforcement learning to find faster small sorting routines in assembly, some of which entered LLVM's libc++.
- **Trust, but verify.** *Alive2* checks that LLVM optimizations are correct (translation validation); *CompCert* is a C compiler with a machine-checked proof that the generated code matches the source semantics. An LLM-proposed optimization needs the same kind of check.

### **For Your Project (AI Policy)**

AI tools are allowed, but the correctness argument must be yours:

1. An LLM may draft a code generator or IR, but **every emitted `.ll` must pass the verifier**.
2. Expected outputs come from your **reference interpreter** or hand computation, **not** from asking the LLM what the program prints.
3. Document one concrete AI mistake and how a test or the verifier exposed it (it is part of the report).

---

## **Project Implementation Patterns**

### **Same Structure in Every Path**

All three paths emit `.ll` text; the generator is a recursive function returning an operand. A **complete, tested** mini compiler for the course's example language is in the example folder (`fplc_mini.py`, about 270 lines of Python: lexer, parser, type checker, code generator). **Read it next to this lecture; it is not a reference solution** (no error recovery, minimal diagnostics).

### **Path B: Python / PLY** (excerpt from `fplc_mini.py`)

```python
def gen(self, e, env):               # returns an LLVM operand: a constant or a %name
    k = e[0]
    if k == "int":  return str(e[1])
    if k == "var":
        if e[1] in env: return env[e[1]]        # let-bound: an SSA value, no memory
        return f"@{e[1]}"                       # a top-level function used as a value
    if k == "let":
        v = self.gen(e[2], env)
        return self.gen(e[3], {**env, e[1]: v}) # extend the environment: that is all
    if k == "if":
        c = self.gen(e[1], env)
        th, el, jn = self.lab("then"), self.lab("else"), self.lab("join")
        self.emit(f"br i1 {c}, label %{th}, label %{el}")
        self.start(th); a = self.gen(e[2], env); th_end = self.cur; self.br(jn)
        self.start(el); b = self.gen(e[3], env); el_end = self.cur; self.br(jn)
        self.start(jn); t = self.tmp()          # th_end / el_end, NOT th / el
        self.emit(f"{t} = phi {llty(e[4])} [ {a}, %{th_end} ], [ {b}, %{el_end} ]")
        return t
    ...
```

### **Path A: C / Bison** (excerpt, tested with gcc and clang)

```c
static const char *gen(Node *n, Env *env) {
    switch (n->kind) {
    case N_LET: {                                /* no instruction: just extend the environment */
        Env e = { n->name, gen(n->a, env), env };
        return gen(n->b, &e);
    }
    case N_IF: {
        const char *c = gen(n->a, env);
        char *th = fresh("then"), *el = fresh("else"), *jn = fresh("join");
        printf("  br i1 %s, label %%%s, label %%%s\n", c, th, el);
        char th_end[32], el_end[32];
        start(th); const char *x = gen(n->b, env); strcpy(th_end, cur); printf("  br label %%%s\n", jn);
        start(el); const char *y = gen(n->c, env); strcpy(el_end, cur); printf("  br label %%%s\n", jn);
        start(jn); char *t = fresh("%t");
        printf("  %s = phi i32 [ %s, %%%s ], [ %s, %%%s ]\n", t, x, th_end, y, el_end);
        return t;
    }
    /* N_INT, N_VAR, N_BIN ... */
    }
}
```

A complete version (`c_snippet/emit.c`) generates IR for nested `if` and `let` and was verified with `opt` and run (the nested `if` produces the phi that names the inner join, as discussed above).

### **Path C: OCaml / Menhir** (sketch; not compiled in the lecture environment, check before use)

```ocaml
let counter = ref 0
let fresh base = incr counter; Printf.sprintf "%s.%d" base !counter
let out = Buffer.create 1024
let emit fmt = Printf.bprintf out ("  " ^^ fmt ^^ "\n")
let cur = ref "entry"
let start label = Printf.bprintf out "%s:\n" label; cur := label

let rec gen env = function
  | Int n -> string_of_int n
  | Var x -> List.assoc x env
  | Add (a, b) ->
      let va = gen env a in
      let vb = gen env b in
      let t = fresh "%t" in
      emit "%s = add i32 %s, %s" t va vb; t
  | Let (x, e1, e2) -> let v = gen env e1 in gen ((x, v) :: env) e2
  | If (c, e1, e2) ->
      let vc = gen env c in
      let th = fresh "then" and el = fresh "else" and jn = fresh "join" in
      emit "br i1 %s, label %%%s, label %%%s" vc th el;
      start th; let va = gen env e1 in let th_end = !cur in emit "br label %%%s" jn;
      start el; let vb = gen env e2 in let el_end = !cur in emit "br label %%%s" jn;
      start jn;
      let t = fresh "%t" in
      emit "%s = phi i32 [ %s, %%%s ], [ %s, %%%s ]" t va th_end vb el_end; t
```

The OCaml LLVM bindings also work, but their API changed with opaque pointers (functions such as `build_load` now take an explicit type, e.g. `build_load2`; names differ between LLVM versions), and emitting text avoids version problems.

---

## **Milestones for Phase 3**

1. **Vertical slice:** `print_int 42` through every stage: `declare`, `@main`, call, link with `runtime.c`
2. Integer and boolean **expressions** (arithmetic, comparisons, unary minus)
3. **`let`** (environment of operands) and **`if`** (`br` + `phi`)
4. **Functions and calls**, recursion, mutual recursion (signatures first)
5. **`&&` and `||`** (short circuit with `phi`)
6. **Function values** and lifted lambdas
7. **Division-by-zero** check and the runtime function
8. Test runner: build everything, run all tests, **verify every `.ll`**
9. Bonuses: closure conversion, `musttail`, Hindley-Milner, lists

Try writing the IR **by hand** for your examples first: you will know what the generator has to produce.

---

## **Testing Your Code Generator**

### **What the Project Requires**

- `make test` or `./run_tests.sh` builds the compiler, runs all tests, prints PASS/FAIL
- **Every emitted `.ll` passes `opt -passes=verify`**
- **Do not validate your compiler against itself:** expected outputs are hand-computed or come from a **reference interpreter** over your AST (about 50 lines)
- Descriptive test names (`run_short_circuit_skips_rhs.fpl`)
- Truncated programs never crash the compiler

### **Differential Testing: Three Judges for Every Test**

The example folder's `run_tests.sh` runs each `.fpl` program and compares:

| Judge | What it catches |
|-------|-----------------|
| compiled with `clang -O0`, against the hand-written `.expected` | code generation bugs |
| the same IR compiled with `clang -O2`, against `.expected` | undefined behavior in your IR that the optimizer exploits (for example a missing division check, a wrong `nsw`) |
| the **reference interpreter** (`interp.py`) | disagreement between two independent implementations of the semantics |

```
PASS bool_eq    PASS div_zero    PASS example    PASS lambda    PASS mutual
PASS nested_if  PASS shadow      PASS shortcircuit    PASS wrap
9 passed, 0 failed
```

The tests cover precedence, shadowing, recursion, mutual recursion, a lambda passed to a function, short-circuit with side effects, wrap-around, and division by zero (exit code 1, message on stderr).

### **Would Your Tests Catch a Wrong Rule? Mutation Check**

In the final presentation the instructor may change something in your code generator (for example `add` to `sub`). Try it yourself:

```
# in the generator: make "+" emit "sub" instead of "add"
FAIL lambda (-O0): rc=0 got: 63           (expected 101)
FAIL shadow (-O0): rc=0 got: -20          (expected 22)
FAIL wrap   (-O0): rc=0 got: 2147483646   (expected -2147483648)
6 passed, 3 failed
```

Three of nine tests fail: the suite notices. If nothing fails after such a change, your suite has a gap.

### **Golden Tests and Their Limit**

Comparing your `.ll` text with a stored file ("golden test") is easy but **brittle**: renaming a temporary breaks it. Prefer tests that **run the program** and compare the output; keep a few golden tests for the AST and typed-AST dumps, where the format is fixed by the spec.

---

## **Common Pitfalls and Solutions**

### **Code Generation Challenges**

| Problem | Solution | Example |
|---------|----------|---------|
| Wrong `phi` predecessors | Record the block you are **in** after generating each branch | `th_end`, not `th` (nested `if`, `&&`) |
| Missing terminator | Every block ends with `br`, `ret` or `unreachable` | after `fpl_div_zero()` write `unreachable` |
| Non-unique names or labels | One counter per function | `then.4`, `join.6` |
| `bool` vs `int` | `i1` for booleans, `zext` for the C runtime | `print_bool` |
| `&&` as `and` | Branch and `phi`: the right side must not run | `false && say 1` |
| Undefined behavior | no `nsw`; check division by zero before `sdiv` | `wrap.fpl`, `div_zero.fpl` |
| Value used outside its branch | Use only values defined on all paths (that is what `phi` is for) | verifier: does not dominate |
| (Stack-machine path) stack imbalance | Every template leaves the stack as it found it | count pushes and pops |
| (Stack-machine path) wrong operand order for `-` | left in `$t1`, right in `$a0`: `sub $a0, $t1, $a0` | `a - b` |

### **Debugging Generated Code**

1. **Print the IR** and read it; comment your generator's output (`; gen for: if n <= 1`).
2. **Run the verifier first;** the message names the line.
3. **Shrink:** reduce a failing program to the smallest one that still fails.
4. **Compare with clang:** write the equivalent C, run `clang -O0 -S -emit-llvm`, compare the structure.
5. **Run with `-O0` and `-O2`:** if they differ, suspect undefined behavior in your IR.
6. For MIPS: step through in MARS or SPIM and watch `$sp`, `$fp`, `$a0`.

---

## **Practical Exercises**

### **Do by Hand**

1. Write LLVM IR for `let x = 5 in x + 1` as the body of `define i32 @f()`.
2. Write the IR for `if a < b then a else b` with `br` and `phi`. (Is `select` also possible? When would it be wrong?)
3. Find the bug:
   ```llvm
   define i32 @f(i1 %a, i1 %b) {
   entry:
     br i1 %a, label %then, label %else
   then:
     br i1 %b, label %t2, label %e2
   t2:
     br label %join2
   e2:
     br label %join2
   join2:
     %x = phi i32 [ 1, %t2 ], [ 2, %e2 ]
     br label %join
   else:
     br label %join
   join:
     %y = phi i32 [ %x, %then ], [ 3, %else ]
     ret i32 %y
   }
   ```
4. Generate IR for `a && (x / y == 2)`. Which block does the division belong to, and why can it not be evaluated before the branch?
5. Stack machine: trace `(7 + 5) * 2` and then `let x = 4 in x * x` (stack contents after every instruction).

### **Answers**

1. `%t = add i32 5, 1` then `ret i32 %t` (`x` is the constant 5).
2. `icmp slt`, `br`, two blocks, `phi`. `select i1 %c, i32 %a, i32 %b` is fine here (both sides are already computed and have no side effects) but **wrong** when a branch has side effects or may fault (calls, division): `select` evaluates both.
3. The `phi` in `join` must name `%join2`, not `%then`: the predecessor of `join` on that path is the **inner join**. Fix: `phi i32 [ %x, %join2 ], [ 3, %else ]`. (The verifier says the entries do not match the predecessors.)
4. The division sits in the `rhs` block (after `br i1 %a, label %rhs, label %join`): if `a` is false it must not run.
5. `[7] [7,5] [12] [12,2] [24]`; and for `let`: push 4 (that is `x`), push `x`, push `x`, multiply, drop `x`.

### **Use an LLM, then Judge It**

Ask an LLM to write the LLVM IR for `fact`, or a small generator for your language. Then: run the verifier; run the tests; classify each failure (syntax, type, logic, undefined behavior); feed the verifier message back once. How often does one repair step fix it? This is exactly the experiment in your final project report.

---

## **Key Takeaways**

1. **An IR separates what you mean from where it runs:** LLVM gives you optimization, register allocation, and many CPUs.
2. **SSA names every value once;** `let` costs nothing, and `phi` merges values where control flow joins.
3. **One template per AST node:** `gen(e, env)` returns an operand, emits instructions, and tracks the current block.
4. **`if` and `&&`/`||` are control flow:** `br` plus `phi`, with the predecessor being the block you are *in*.
5. **Avoid undefined behavior** in your IR: no `nsw` when the language wraps, a check before `sdiv`.
6. **The verifier and differential tests are your safety net:** verify, run at `-O0` and `-O2`, compare with a reference interpreter.
7. **The hand-written stack machine and MIPS code** show what a back end does: frames, calling sequence, registers.
8. **AI changes how we write generators, not how we trust them:** verify, test, document.

### **Modern Relevance**

- **LLVM and MLIR** are the base of many compilers (Clang, Rust, Swift, ML compilers)
- **WebAssembly:** a new portable target
- **JIT compilation:** run-time code generation
- **ML compilers:** code generation for GPUs and accelerators
- **Formal verification:** proving generated code correct

---

## **Next Steps and Resources**

**Advanced topics (next weeks)**
- Register allocation (graph coloring, linear scan)
- Instruction selection (tree pattern matching)
- Peephole and global optimization, SSA construction
- Garbage collection support in code generators

**Official documentation and tools**
- LLVM Language Reference: https://llvm.org/docs/LangRef.html
- LLVM Kaleidoscope tutorial: https://llvm.org/docs/tutorial/
- LLVM opaque pointers: https://llvm.org/docs/OpaquePointers.html
- MLIR: https://mlir.llvm.org · OpenXLA: https://openxla.org/xla · Triton: https://triton-lang.org
- Compiler Explorer (see IR and assembly side by side): https://godbolt.org
- SPIM and MARS (MIPS simulators) · RARS (RISC-V)

**Books and papers**
- Cooper and Torczon, *Engineering a Compiler*
- Appel, *Modern Compiler Implementation*
- Muchnick, *Advanced Compiler Design and Implementation*
- Nystrom, *Crafting Interpreters* (free online), https://craftinginterpreters.com
- Mankowitz et al., *Faster sorting algorithms discovered using deep reinforcement learning*, Nature 2023
- Cummins et al., *Meta Large Language Model Compiler*, https://arxiv.org/abs/2407.02524
