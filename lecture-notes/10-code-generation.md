## **Code Generation - From AST to Machine Code**

### **The Final Compiler Phase**

```
Frontend Complete:           Backend:
AST + Symbol Table    →    Target Code
Type Information      →    Executable Program
Semantic Checks       →    Runtime Efficiency
```

### **Key Challenge:**
**How do we translate our high-level program understanding into low-level machine instructions?**

---

## **Intermediate Representations**
- Graphical IRs 
  - syntax related trees: AST, Parse Tree
  - Graphs: control-flow graph, dependency graph (DAG), call graph etc
- Linear IRs
  - stack machine code
  - one address, two-address, three address code (t5 ← t4 - t3)

---

## **Code Generation Strategies**
### **Three Major Approaches**

**1. Stack-Based (Interpreters/Virtual Machines)**
```java
// Java Bytecode-like
push 5
push 3
add
store result
```

---

**2. Register-Based (Traditional Compilers)**
```assembly
# x86-64
mov rax, 5
add rax, 3
mov [result], rax
```

---

**3. Modern SSA Form (LLVM, GCC)**: *Static Single Assignment*: every name is assigned exactly once
```llvm
; LLVM IR
%1 = add i32 5, 3
store i32 %1, ptr @result      ; LLVM 15+ pointers are untyped: ptr
```
---


### **Why Start with Stack Machines?**
- **Simpler** to understand and implement
- **Uniform** code generation pattern
- **Educational** foundation for more complex approaches

---

## **Stack Machine Fundamentals**

### **The Simple Evaluation Model**

**Basic Operations:**
- `push value` - push value onto stack
- `pop` - remove top value
- `add` - pop two values, push their sum
- `sub` - pop two values, push their difference

---

**Example: `(7 + 5) * 2`**
```
push 7      [7]
push 5      [7, 5]
add         [12]
push 2      [12, 2]
multiply    [24]
```

---

### **Real-World Usage:**
- **Java Bytecode** (JVM)
- **WebAssembly**
- **Python Bytecode**
- **.NET CIL**

---

## **From Stack Machine to Real Hardware**

---

## **Introduction to MIPS Architecture**

### **Why MIPS for Teaching Code Generation?**

**MIPS (Microprocessor without Interlocked Pipeline Stages)**
- **Widely used for teaching**: Clean, regular instruction set
- **RISC architecture**: Reduced Instruction Set Computer
- **Load-store model**: Clear separation between memory and ALU operations
- **Simple addressing modes**: Easy to understand and generate code for

---

### **MIPS vs x86 NASM Comparison**

| Aspect | **MIPS (Our Focus)** | **x86 NASM (Common Alternative)** |
|--------|----------------------|-----------------------------------|
| **Architecture** | RISC | CISC |
| **Registers** | 32 general-purpose ($0-$31) | 16 general-purpose (rax, rbx, ...) |
| **Instruction Format** | Fixed 32-bit, regular | Variable length, complex |
| **Memory Access** | Load/Store only (separate instructions) | Memory operands in most instructions |
| **Calling Convention** | Clear register usage ($a0-$a3 for args, $v0-$v1 for return) | Register-based: rdi, rsi, ... (System V) or rcx, rdx, ... (Windows) |
| **Teaching Value** | **Excellent** - simple, predictable | Complex, historical quirks |

---

### **MIPS Registers You Need to Know:**
```mips
# Important MIPS registers for our compiler
$a0-$a3:    # Argument registers (first 4 function args)
$v0-$v1:    # Return value registers
$t0-$t9:    # Temporary registers (caller-saved)
$s0-$s7:    # Saved registers (callee-saved)
$gp:        # Global pointer
$sp:        # Stack pointer
$fp:        # Frame pointer
$ra:        # Return address
```
[mips summary](https://www.cs.tufts.edu/comp/140/lectures/Day_3/mips_summary.pdf)  

---

### **MIPS Instruction Examples:**
```mips
# Load/Store
lw $t0, 4($sp)      # Load word from memory (stack+4) into $t0
sw $a0, 0($sp)      # Store word from $a0 to memory (stack)

# Arithmetic
add $a0, $t1, $t2   # $a0 = $t1 + $t2
sub $a0, $t1, $t2   # $a0 = $t1 - $t2
mul $a0, $t1, $t2   # $a0 = $t1 * $t2

# Control flow
beq $t0, $t1, label # Branch if equal
j label             # Unconditional jump
jal func            # Jump and link (call function)
jr $ra              # Jump to return address

# Immediate operations
addi $t0, $t1, 5    # $t0 = $t1 + 5
li $t0, 42          # Load immediate: $t0 = 42 (pseudo-instruction: li, move, b, sgt, ... expand to real ones)
```

---


### **Example: Simple Expression `a + b * c`**

**AST Representation:**
```
      +
     / \
    a   *
       / \
      b   c
```
**MIPS Code Generation:**
```mips
# Assuming: a in $s0, b in $s1, c in $s2
# Result in $a0

# Compute b * c
mul $t0, $s1, $s2   # $t0 = b * c

# Compute a + (b * c)
add $a0, $s0, $t0   # $a0 = a + $t0
```
---



**Stack Machine Approach `a + b * c` (for comparison):**
```
push a      # [a]
push b      # [a, b]
push c      # [a, b, c]
multiply    # [a, (b*c)]
add         # [a+(b*c)]
```

---

### **Simulating Stack on MIPS**

---
**Stack Machine:** `push 7`
```mips
li $a0, 7       # Load 7 into accumulator
sw $a0, 0($sp)  # Store accumulator to stack
addiu $sp, $sp, -4  # Move stack pointer down
```

**Register Mapping:**
- **Accumulator**: `$a0` - holds current computation result
- **Stack Pointer**: `$sp` - points to top of stack
- **Temporary**: `$t1` - for intermediate operations
- **Frame Pointer**: `$fp` - for stable variable access

---

 **Example: Variable Assignment `x = y + 5`**

**AST:**
```
    =
   / \
  x   +
     / \
    y   5
```

**MIPS Code Generation with Stack:`x = y + 5`**
```mips
# Assuming y's value is in memory at offset 8 from $fp

# Load y
lw $a0, 8($fp)      # $a0 = y
sw $a0, 0($sp)      # Push y onto stack
addiu $sp, $sp, -4

# Load 5
li $a0, 5           # $a0 = 5

# Add
lw $t1, 4($sp)      # Pop y into $t1
add $a0, $t1, $a0   # $a0 = y + 5
addiu $sp, $sp, 4   # Adjust stack

# Store result into x (offset 4 from $fp)
sw $a0, 4($fp)      # x = $a0
```

---


## **let in a Functional Language**

### **Names Map to Frame Offsets**

```
cgen(let x = e1 in e2):  cgen(e1)            # value in $a0
                         push $a0            # x lives in the slot just pushed
                         cgen(e2)            # with env + { x -> that slot }
                         pop (drop x)        # result stays in $a0
```

- Week 8: the environment mapped **names to types**. Here it maps **names to offsets** from `$fp`: parameters at `4($fp)`, `8($fp)`, ... and let-bound values at `-4($fp)`, `-8($fp)`, ...
- Same scoping and shadowing rules, different payload.
- Compare with LLVM IR (next slides): `let x = y + 5 in x * 2` is just `%x = add i32 %y, 5` and `%r = mul i32 %x, 2`. No memory, no stack slot.

---

## **Code Generation for Expressions**

### **Recursive AST Traversal**

```python
def cgen(node):
    if node.type == 'number':
        return f"li $a0, {node.value}"
    
    elif node.type == 'binary_op':
        left_code = cgen(node.left)
        right_code = cgen(node.right)
        
        if node.operator == '+':
            return f"""
            {left_code}
            sw $a0, 0($sp)
            addiu $sp, $sp, -4
            {right_code}
            lw $t1, 4($sp)
            add $a0, $t1, $a0
            addiu $sp, $sp, 4
            """
        raise NotImplementedError(node.operator)   # never return None silently
```

### **Example: `3 + (7 + 5)`**
```mips
# 3 + (7 + 5)
li $a0, 3
sw $a0, 0($sp)
addiu $sp, $sp, -4

li $a0, 7
sw $a0, 0($sp)
addiu $sp, $sp, -4

li $a0, 5
lw $t1, 4($sp)
add $a0, $t1, $a0
addiu $sp, $sp, 4

lw $t1, 4($sp)
add $a0, $t1, $a0
addiu $sp, $sp, 4
```

---

## **Control Flow Generation**

### **Conditionals and Loops**

**If-Then-Else Pattern:**
```mips
cgen(condition)
beq $a0, $zero, false_branch

true_branch:
cgen(true_expression)
b end_if

false_branch:
cgen(false_expression)

end_if:
# Continue execution...
```

### **Example 3: If-Statement**
```c
// C code
if (x > 0) {
    y = 10;
} else {
    y = 20;
}
```
**MIPS Code Generation:**
```mips
# Assume x is at offset 8($fp), y at offset 12($fp)

# Load x
lw $t0, 8($fp)      # $t0 = x

# Compare x > 0
sgt $t1, $t0, $zero # $t1 = (x > 0) ? 1 : 0
beq $t1, $zero, else_label  # Branch if false

# Then part: y = 10
li $t2, 10
sw $t2, 12($fp)     # y = 10
j end_if            # Skip else part

else_label:
# Else part: y = 20
li $t2, 20
sw $t2, 12($fp)     # y = 20

end_if:
# Continue with next statements...
```

---

**While Loop Pattern:**
```mips
loop_start:
cgen(condition)
beq $a0, $zero, loop_end

cgen(loop_body)
b loop_start

loop_end:
# Continue after loop
```

---

## **Function Call Implementation**

### **The Calling Convention**

**Caller Responsibilities:**
1. Save frame pointer
2. Push arguments (reverse order)
3. Jump to the function (`jal` puts the return address in `$ra`)

**Callee Responsibilities:**
1. Set the new frame pointer and save `$ra` on the stack
2. Execute the function body (result in `$a0`)
3. Restore `$ra`, pop the arguments and the saved frame pointer
4. Restore the caller's frame pointer, return

---

```mips
# Caller: preparing f(x, y)
sw $fp, 0($sp)       # Save frame pointer
addiu $sp, $sp, -4

cgen(y)              # Push second argument
sw $a0, 0($sp)
addiu $sp, $sp, -4

cgen(x)              # Push first argument  
sw $a0, 0($sp)
addiu $sp, $sp, -4

jal f_entry          # Jump to function
```

---

## **The Callee: Prologue and Epilogue**

```mips
f_entry:                      # f has n = 2 arguments
  move  $fp, $sp              # $fp points at the (empty) slot for $ra
  sw    $ra, 0($sp)           # save the return address
  addiu $sp, $sp, -4
  # ... body, result in $a0 ...
  lw    $ra, 4($sp)           # reload the return address
  addiu $sp, $sp, 16          # pop $ra, x, y and the saved $fp slot (4*n + 8)
  lw    $fp, 0($sp)           # restore the caller's frame pointer
  jr    $ra
```

---

## **Activation Records Revisited**

---

### **Connecting Runtime with Code Generation**

**Complete Activation Record for `f(x, y)`:**

```
High Address
┌─────────────┐
│  Old FP     │  ← $fp + 12  (pushed by the caller)
├─────────────┤
│      y      │  ← $fp + 8
├─────────────┤
│      x      │  ← $fp + 4
├─────────────┤
│ Return Addr │  ← $fp points here (pushed by the callee)
├─────────────┤
│ let vars,   │  ← $fp - 4, $fp - 8, ...
│ temporaries │
└─────────────┘ ← $sp
Low Address
```
---

### **Variable Access:**
```mips
# Access parameter x (offset +4 from $fp)
lw $a0, 4($fp)

# Access parameter y (offset +8 from $fp)  
lw $a0, 8($fp)
```

---

## **Object-Oriented Code Generation**

### **From Classes to Machine Structures**

**Class Definition:**
```java
class Point {
    int x, y;
    void move(int dx, int dy) {
        x = x + dx;
        y = y + dy;
    }
}
```
---

**Memory Layout:**
```
Point Object:
┌─────────────┐
│  Class Tag  │  // Runtime type info
├─────────────┤
│ Object Size │  // For garbage collection  
├─────────────┤
│ Dispatch Ptr│  // Virtual method table
├─────────────┤
│      x      │  // Instance variable
├─────────────┤
│      y      │  // Instance variable
└─────────────┘
```

---

## **Dynamic Dispatch Implementation**

### **Virtual Method Tables**

**Class Hierarchy:**
```
     Animal
     /    \
   Dog    Cat
```
---

**Virtual Method Tables:**
```
Animal VTable    Dog VTable      Cat VTable
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  speak()    │  │  speak()    │  │  speak()    │
│  eat()      │  │  eat()      │  │  eat()      │
│  sleep()    │  │  sleep()    │  │  sleep()    │
└─────────────┘  │  bark()     │  │  meow()     │
                 └─────────────┘  └─────────────┘
```
---

**Method Call Implementation:**
```mips
# animal.speak()
lw $t0, 8($a0)      # Load vtable pointer from object
lw $t1, 0($t0)      # Load speak() method address
jalr $t1            # Jump to method
```

---

## **Modern Code Generation Trends**

### **Beyond Simple Stack Machines**

**Static Single Assignment (SSA):**
```llvm
; LLVM IR - Explicit data flow
%1 = add i32 %x, %y
%2 = mul i32 %1, 2
ret i32 %2
```

---

**Just-In-Time (JIT) Compilation:**
- **Profile-guided optimization**: Optimize based on runtime behavior
- **Deoptimization**: Fall back to interpreter for unexpected cases
- **Tiered compilation**: Start fast, optimize hot code paths

---

**WebAssembly:**
```wat
;; WebAssembly text format
(func $add (param $x i32) (param $y i32) (result i32)
  local.get $x
  local.get $y
  i32.add)
```

---

## **ML Compilers and AI**

### **Code Generation Beyond CPUs**

- Deep learning frameworks are compilers: a model is a **graph of tensor operations** (the IR), optimized and lowered to GPU code.
- **MLIR** (multi-level IR in the LLVM project), **XLA** (JAX, TensorFlow), **Apache TVM**, **Triton** (GPU kernels), **`torch.compile`** (TorchInductor uses Triton on GPUs).
- Same pipeline as this course: front end, typed IR (tensor shapes, week 8), optimization (fusion, memory planning), code generation for a different target.

### **LLMs and Generated Code**

- Meta's *LLM Compiler* is trained on LLVM IR and assembly; DeepMind's AlphaDev found faster small sorting routines.
- **Trust but verify:** *Alive2* checks LLVM optimizations, *CompCert* is a compiler with a machine-checked correctness proof. LLM-written IR needs the verifier, tests and a reference interpreter.

---

## **Project Target: Textual LLVM IR**

### **The Pipeline of Phase 3**

```
prog.fpl -> lexer/parser -> AST -> type checker -> typed AST -> code generator -> prog.ll
prog.ll  -> opt (optional) -> llc -> assembly -> linker -> executable   (clang drives these)
```

```bash
./fplc prog.fpl --emit-ll -o prog.ll
opt -passes=verify -disable-output prog.ll     # is the IR well formed?
clang prog.ll runtime.c -o prog && ./prog
```

- **Emit readable `.ll` text yourself.** Use LLVM 15+ (opaque `ptr`).
- LLVM does what the MIPS part of this lecture does by hand: instruction selection, register allocation, frames, for x86-64, ARM64, RISC-V.
- A tested mini compiler for the example language: `10-code-generation-example/`.

---

### **Anatomy of an LLVM IR File**

```llvm
declare i32 @print_int(i32)                 ; provided by runtime.c

define i32 @double(i32 %x) {                ; return type, @name, typed parameters
entry:                                      ; a basic block: label ...
  %t = mul i32 %x, 2                        ; ... instructions: %name = opcode type operands
  ret i32 %t                                ; ... and exactly ONE terminator (ret, br)
}
```

- `@name` global, `%name` local. Types are explicit: `i32`, `i1` (bool), `ptr`.
- **SSA:** each `%name` is defined once. Functions may be called before they are defined (mutual recursion works).
- Prefer named temporaries (`%t.1`): unnamed ones must be numbered in order.

---

### **From Your Language to LLVM IR**

| Your language | LLVM IR |
|---------------|---------|
| `int` (32-bit, wraps) | `i32`, plain `add`, `sub`, `mul` (no `nsw`) |
| `bool` | `i1`; `zext i1 %b to i32` for the C runtime |
| function type | `ptr` |
| `let x = e1 in e2` | **no instruction**: bind `x` to the value of `e1` |
| `e1 < e2` | `icmp slt` (result `i1`) |
| `if c then a else b` | `br i1`, two blocks, join block with `phi` |
| `a && b`, `a \|\| b` | `br` (short circuit) + `phi` |
| `e1 / e2` | zero check, then `sdiv` |
| call `f a b` / function value `f x` | `call i32 @f(...)` / `call i32 %f(i32 %x)` |
| `fun (x: int) -> e` | lifted top-level function, value `ptr @lambda.1` |

```
gen(e, env)   emits instructions for e and RETURNS the operand holding its value
env           maps names to LLVM operands   (week 8: names -> types; now: names -> values)
```

---

### **let and if**

```
gen(Let(x, e1, e2), env) = v = gen(e1, env);  return gen(e2, env + {x -> v})
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

**The classic bug:** a branch may contain another `if`, so the block you are **in** at its end is not the one you started. The `phi` must name the block where the branch to `join` is emitted (`then_end`), not `then`.

```
PHI node entries do not match predecessors!      <- what the verifier says
```

---

### **Short-Circuit, Division, Function Values**

```llvm
; a && b : the right operand may not run (it can print or fail)
entry:
  br i1 %a, label %rhs, label %join
rhs:
  ; ... code for b, result %b ...
  br label %join
join:
  %t = phi i1 [ false, %entry ], [ %b, %rhs ]
```

```llvm
; e1 / e2 : sdiv by zero is undefined behavior, so check first
  %z = icmp eq i32 %b, 0
  br i1 %z, label %divzero, label %divok
divzero:
  call void @fpl_div_zero()      ; runtime.c: message on stderr, exit(1)
  unreachable
divok:
  %q = sdiv i32 %a, %b
```

```llvm
define i32 @apply(ptr %f, i32 %x) {
  %t = call i32 %f(i32 %x)             ; indirect call through a ptr
  ret i32 %t
}
  %r = call i32 @apply(ptr @double, i32 5)
```

A lambda that captures nothing is lifted to a new top-level function; capturing needs a closure (bonus).

---

## **Optimization Techniques**

### **From Naive to Efficient Code Generation**

**Naive Stack Code:**
```mips
li $a0, 5
sw $a0, 0($sp)    # Unnecessary store/load
addiu $sp, $sp, -4
li $a0, 3
lw $t1, 4($sp)
add $a0, $t1, $a0
addiu $sp, $sp, 4
```
---

**Optimized Register Code:**
```mips
li $t0, 5         # Keep in register
li $a0, 3
add $a0, $t0, $a0 # Direct computation
```
---

**Common Optimizations:**
- **Register allocation**: Minimize memory traffic
- **Common subexpression elimination**: Reuse computed values
- **Constant folding**: Compute constants at compile time
- **Dead code elimination**: Remove unused computations

---

## **What opt and llc Do to Your IR**

`opt -passes='default<O2>'` on the example program (real output, trimmed): `main` becomes `tail call i32 @print_int(i32 10)` (constant folding and inlining), and `fact` becomes a loop (tail recursion elimination):

```llvm
else.3:
  %n.tr3 = phi i32 [ %t.5, %else.3 ], [ %n, %entry ]
  %accumulator.tr2 = phi i32 [ %t.7, %else.3 ], [ 1, %entry ]
  ...
```

The same IR for `x * y + 10` through `llc -O2`:

```asm
# x86-64                   # AArch64                  # RISC-V                # MIPS32
imull %esi, %edi           mov  w8, #10                mul   a0, a0, a1        mul   $1, $4, $5
leal  10(%rdi), %eax       madd w0, w0, w1, w8         addiw a0, a0, 10        jr    $ra
retq                       ret                         ret                     addiu $2, $1, 10   # delay slot
```

You emit IR; LLVM provides optimization, register allocation and the target ABI.

---

## **Real-World Code Generation Examples**

### **Industry Compiler Output**

**GCC for Simple C:**
```c
// C source
int calculate(int x, int y) {
    return x * y + 42;
}
```

```assembly
; GCC x86-64 output (optimized)
calculate:
    mov eax, edi
    imul eax, esi
    add eax, 42
    ret
```

---

**LLVM for the Same Function:**
```llvm
; LLVM IR
define i32 @calculate(i32 %x, i32 %y) {
  %1 = mul i32 %x, %y
  %2 = add i32 %1, 42
  ret i32 %2
}
```

---

## **Project Path Guidance - Code Generation**

The final project target is **LLVM IR** (text). Combine the lecture examples (`5-bison-example`, `8-semantic-analysis-type-checking`) with the rules above. The hand-written MIPS generator is the classic alternative and shows what LLVM does for you.

### **Same Structure in Every Path**

**C / Flex / Bison:**
```c
static const char *gen(Node *n, Env *env) {
    switch (n->kind) {
    case N_LET: { Env e = { n->name, gen(n->a, env), env }; return gen(n->b, &e); }
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

---

**Python / PLY:**
```python
def gen(self, e, env):               # returns an operand: a constant or a %name
    k = e[0]
    if k == "int": return str(e[1])
    if k == "var": return env[e[1]]
    if k == "let":
        v = self.gen(e[2], env)
        return self.gen(e[3], {**env, e[1]: v})
    if k == "if":
        c = self.gen(e[1], env)
        th, el, jn = self.lab("then"), self.lab("else"), self.lab("join")
        self.emit(f"br i1 {c}, label %{th}, label %{el}")
        self.start(th); a = self.gen(e[2], env); th_end = self.cur; self.br(jn)
        self.start(el); b = self.gen(e[3], env); el_end = self.cur; self.br(jn)
        self.start(jn); t = self.tmp()
        self.emit(f"{t} = phi i32 [ {a}, %{th_end} ], [ {b}, %{el_end} ]")
        return t
```

---

**OCaml / Menhir** (sketch, not compiled in the lecture environment: check before use):
```ocaml
let rec gen env = function
  | Int n -> string_of_int n
  | Var x -> List.assoc x env
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
  | _ -> failwith "..."
```
The OCaml LLVM bindings also work, but their API changed with opaque pointers (e.g. `build_load2` takes an explicit type); emitting text avoids version problems.

---

**Milestones:**
1. `print_int 42` through every stage (declare, `@main`, call, link `runtime.c`)
2. Integer and boolean expressions
3. `let` and `if` (`br` + `phi`)
4. Functions, calls, recursion, mutual recursion (signatures first)
5. `&&` and `||` (short circuit), function values, division-by-zero check
6. Test runner: build, run everything, **verify every `.ll`**

---

### **The Verifier Is Your Best Friend**

```bash
opt -passes=verify -disable-output prog.ll      # silent when the IR is well formed
```

| Message | Usual cause |
|---------|-------------|
| `PHI node entries do not match predecessors` | wrong block names in `phi` (nested `if`, `&&`) |
| `Instruction does not dominate all uses` | using a value defined inside a branch after the join |
| `defined with type 'i1' but expected 'i32'` | `bool` and `int` confused, missing `zext` |
| `expected instruction opcode` | a block without a terminator (`ret`, `br`) |

Tip: write the equivalent C and run `clang -S -emit-llvm -O1` to see idiomatic IR.

---

## **Testing Your Code Generator**

### **Project Requirements**

- One command (`make test` or `./run_tests.sh`) builds the compiler, runs all tests, prints PASS/FAIL
- **Every emitted `.ll` passes `opt -passes=verify`**
- **Do not validate your compiler against itself:** expected outputs are hand-computed or come from a **reference interpreter** (about 50 lines)

### **Differential Testing: Three Judges**

| Judge | What it catches |
|-------|-----------------|
| compiled with `clang -O0`, against the hand-written expected output | code generation bugs |
| the same IR at `-O2`, against the expected output | undefined behavior in your IR (missing division check, wrong `nsw`) |
| the reference interpreter | disagreement between two implementations of the semantics |

The example folder's `run_tests.sh` does exactly this (9 programs: precedence, shadowing, recursion, short circuit, wrap-around, division by zero, ...).

### **Mutation Check**

The instructor may change one thing in your generator (for example `add` to `sub`). With that change in the example compiler, 3 of 9 tests fail: the suite notices. If nothing fails after such a change, your suite has a gap.

---

## **Common Pitfalls & Solutions**

### **Code Generation Challenges**

| Problem | Solution | Example |
|---------|----------|---------|
| **Stack imbalance** | Count pushes/pops | Add `stack_depth` tracking |
| **Register spilling** | Save/restore temps | Use `$s0-$s7` for long-lived values |
| **Label generation** | Unique label names | Use counter: `label_1`, `label_2` |
| **Function prologue/epilogue** | Standard template | Always save `$fp`, `$ra` |
| **Type mismatches** | Runtime checks or static typing | Add type annotations in AST |
| **Wrong `phi` predecessors** | Record the block you are **in** after each branch | nested `if`, `&&` |
| **`&&` as `and`** | Branch + `phi`: the right side must not run | `false && f 1` |
| **Undefined behavior in IR** | No `nsw` if the language wraps; check before `sdiv` | `wrap`, `div_zero` tests |

### **Debugging Generated Code:**
1. **Add comments** to generated code:
   ```mips
   # Generated for: x = y + 5
   lw $a0, 8($fp)      # Load y
   addi $a0, $a0, 5    # y + 5
   sw $a0, 4($fp)      # Store to x
   ```

2. **Step-by-step simulation**:
   ```python
   def simulate_mips(code):
       registers = {'$a0': 0, '$t0': 0, ...}
       memory = {}
       stack = []
       # Execute each instruction, print state
   ```

3. **Compare with reference**:
   - Use GCC/Clang to compile equivalent C code
   - Compare your MIPS output with compiler's output
   - Learn optimization techniques

---


## **Golden and Execution Tests**

### **Verification Strategies**

**Golden Testing:**
```python
def test_code_generation():
    source = "3 + 5 * 2"
    expected_asm = """
        li $a0, 3
        sw $a0, 0($sp)
        addiu $sp, $sp, -4
        li $a0, 5
        sw $a0, 0($sp)
        addiu $sp, $sp, -4
        li $a0, 2
        lw $t1, 4($sp)
        mul $a0, $t1, $a0
        addiu $sp, $sp, 4
        lw $t1, 4($sp)
        add $a0, $t1, $a0
        addiu $sp, $sp, 4
    """
    assert generate_code(parse(source)) == normalize_asm(expected_asm)
```
---

**Execution Testing:**
```python
def test_execution():
    program = "print_int (3 + 5)"
    asm = generate_code(parse(program))
    result = run_on_simulator(asm)
    assert result == 8
```

---

## **Key Takeaways**

### **Code Generation Principles**

1. **Abstraction Layers**: AST → Intermediate Representation → Target Code
2. **Pattern Matching**: Each AST node type has a code generation template
3. **Resource Management**: Stack vs registers, memory layout decisions
4. **Runtime Integration**: Code must work with your runtime environment design
5. **Testing**: Generated code must be verified for correctness (verifier, `-O0` vs `-O2`, reference interpreter)
6. **LLVM IR** (your project target): SSA, `let` is just a name, `if` is `br` + `phi`, `&&` short-circuits with control flow
7. **AI**: LLM-written code or IR is a proposal; verify and test it

---

### **Modern Relevance:**
- **WebAssembly**: New target for web compilation
- **JIT Compilation**: Runtime code generation and optimization
- **Cross-compilation**: Generating code for different architectures
- **Formal Verification**: Proving correctness of generated code (CompCert, Alive2)
- **ML compilers**: MLIR, XLA, TVM, Triton generate GPU code

---

## **Next Steps & Resources**

### **Continuing Compiler Journey**

**Advanced Topics (next weeks):**
- **Register Allocation** (graph coloring, linear scan)
- **Instruction Selection** (tree pattern matching)
- **Peephole Optimization** (local pattern-based improvements)
- **Vectorization** (SIMD instruction generation)

**Tools & Libraries:**
- **LLVM**: Industrial-strength code generation framework
- **GNU Lightning**: Lightweight JIT compilation library
- **Cranelift**: Modern code generator written in Rust
- **DynASM**: Dynamic assembler for JIT compilers

---

**Further Reading:**
- "Engineering a Compiler" by Cooper & Torczon
- "Advanced Compiler Design and Implementation" by Muchnick
- LLVM Documentation: https://llvm.org/docs/
- LLVM Language Reference: https://llvm.org/docs/LangRef.html
- LLVM Kaleidoscope tutorial: https://llvm.org/docs/tutorial/
- Compiler Explorer: https://godbolt.org

