## **Runtime Environments: Connecting the Front End to the Back End**

BIL 463 Compiler Design · Week 9

### **Where We Are in the Compiler**

```
Source -> Lexer -> Parser -> AST -> Semantic analysis -> IR -> Code generation -> Executable
                                    (types, scopes)             ^
                                                                |
                                     Runtime environment: the contract the generated code runs under
```

- Last week: **what** the program means (types, scopes, symbol tables).
- This week: **how** a running program is laid out in memory: frames, calling conventions, heap.
- Next week: code generation, which has to follow the rules of this week.

### **The Big Question**

**"Now that we know the program is well typed, what happens in memory when it actually runs?"**

By the end of this lecture you can

- draw the memory layout of a process and say what lives where
- draw an activation record (stack frame) **in the right order** and trace a call chain
- explain calling conventions on x86-64, ARM64, RISC-V and MIPS
- explain why recursion uses stack space, and what a **tail call** changes
- compare manual, ownership-based, reference-counted and garbage-collected memory management
- connect all of this to your final project (LLVM IR) and to AI runtimes

---

## **From Static Analysis to Dynamic Execution**

### **Static vs Dynamic World**

| Static (compile time) | Dynamic (run time) |
|---------------------|------------------|
| AST structure | Activation records |
| Types | Object and value layout in memory |
| Symbol tables | Stack frames and register contents |
| Scope rules | Procedure calls and returns |
| **Known before the program runs** | **Determined while it runs** |

### **Key Insight**

The runtime environment bridges the compiler's **static knowledge** (a name `x` of type `int` in scope) and **dynamic execution** (a 4-byte slot at some offset from the frame pointer).

> Week 8 built an environment that maps **names to types**. Code generation builds an environment that maps **names to locations**. Same scoping rules, different payload.

---

## **Memory Layout: The Program's Home**

### **Traditional Process Memory (Linux, x86-64)**

```
High addresses
┌───────────────────────┐
│        Stack          │  function calls, locals; grows DOWN
│          |            │
│          v            │
│                       │
│          ^            │
│          |            │
│        Heap           │  dynamic allocation; grows UP (malloc, new)
├───────────────────────┤
│  Static data (.data,  │  globals, constants
│  .bss)                │
├───────────────────────┤
│  Code (.text)         │  machine instructions, read-only
└───────────────────────┘
Low addresses
```

(Real processes also map shared libraries, thread stacks and memory-mapped files in the gap.)

### **Modern Reality**

- **Virtual memory:** every process sees its own address space; the OS and hardware map pages to RAM.
- **Multiple threads = multiple stacks.** Each thread has its own stack (about 8 MB by default on Linux for the main thread).
- **Security hardening:** ASLR (randomized addresses), non-executable stack (W^X), stack canaries, and hardware shadow stacks (for example Intel CET) that keep a second copy of return addresses.

---

## **Activation Records: The Heart of Execution**

### **What Happens When You Call a Function?**

```c
int multiply(int a, int b) {
    int result = a * b;
    return result;
}

int calculate(void) {
    int x = 5;
    int y = multiply(x, 3);
    return y + 1;
}

int main(void) {
    int final = calculate();
    return final;
}
```

Every call needs a place for: the **arguments**, the **return address**, the **caller's frame pointer** (so the caller can continue), the **locals**, and **temporaries**. That block of memory is the **activation record** (stack frame). It is created on call and destroyed on return, so calls and returns nest like parentheses: a **stack**.

### **The Contents of a Frame**

| Component | Purpose | Example |
|-----------|---------|---------|
| **Arguments** | Values passed by the caller (registers first, extra ones on the stack) | `a=5, b=3` |
| **Return address** | Where to continue in the caller | an address inside `calculate` |
| **Control link** (saved frame pointer) | The caller's frame, so it can be restored | the caller's `rbp` |
| **Locals and spills** | Variables and values that did not fit in registers | `result` |
| **Temporaries** | Intermediate results of expressions | `a * b` before it is stored |

### **A Frame, Drawn Correctly**

The stack grows toward **lower** addresses. The caller's data is **above** (higher), the callee's data is **below**:

```
High address
┌──────────────────────────┐
│ arguments passed on the  │  (only those that did not fit in registers)
│ stack, pushed by caller  │
├──────────────────────────┤
│ return address           │  pushed by the `call` instruction
├──────────────────────────┤
│ saved frame pointer      │  <- FP points here (control link)
├──────────────────────────┤
│ locals and spilled regs  │  FP - 4, FP - 8, ...
├──────────────────────────┤
│ temporaries              │  <- SP points here (top of stack)
└──────────────────────────┘
Low address
```

Order, from high to low address: **arguments, return address, saved FP, locals, temporaries.** Locals sit at **negative** offsets from FP, arguments passed on the stack at **positive** offsets.

---

## **A Real Frame: x86-64 and gcc**

### **Same Function, Two Optimization Levels**

```c
int calculate(int x, int y) {
    int result = x * y;
    return result + 10;
}
```

`gcc -O0 -S -masm=intel` (gcc 13, Linux, simplified):

```asm
calculate:
    push rbp               ; save the caller's frame pointer
    mov  rbp, rsp          ; the new frame pointer
    mov  DWORD PTR -20[rbp], edi   ; x arrives in edi: spill it
    mov  DWORD PTR -24[rbp], esi   ; y arrives in esi: spill it
    mov  eax, DWORD PTR -20[rbp]
    imul eax, DWORD PTR -24[rbp]
    mov  DWORD PTR -4[rbp], eax    ; result lives at rbp-4
    mov  eax, DWORD PTR -4[rbp]
    add  eax, 10
    pop  rbp
    ret                    ; return value is in eax
```

`gcc -O2`:

```asm
calculate:
    imul edi, esi          ; no frame at all
    lea  eax, 10[rdi]      ; result in eax
    ret
```

**A frame is an implementation choice, not a law.** With optimization the compiler keeps everything in registers, and `main` calling `calculate(5, 3)` is even folded to the constant 25. Try it yourself on https://godbolt.org.

> The locals are at **negative** offsets from `rbp`, below the saved `rbp` and return address. Registers `edi` and `esi` are the first two argument registers of the System V ABI (Linux, macOS). Windows uses different ones (next slides).

### **Frame Pointers Are Back**

For years compilers omitted the frame pointer (`-fomit-frame-pointer`) to free one register. Recent Fedora and Ubuntu 24.04 LTS **enable frame pointers by default again**, because profilers and tracers walk the chain of saved frame pointers to build stack traces. A runtime design choice driven by tooling.

---

## **Calling Conventions (ABIs)**

### **Who Puts the Arguments Where?**

A **calling convention** is the agreement between caller and callee: which registers hold arguments, who saves which registers, who cleans the stack. Compilers, libraries and the OS must all agree; this is part of the **ABI**.

| Platform | Integer arguments | Return value | Notes |
|----------|-------------------|--------------|-------|
| **x86-64 System V** (Linux, macOS) | `rdi, rsi, rdx, rcx, r8, r9`, then stack | `rax` | 128-byte red zone below `rsp` for leaf functions |
| **x86-64 Windows** | `rcx, rdx, r8, r9`, then stack | `rax` | caller reserves **shadow space** for the four register arguments |
| **ARM64 (AAPCS64)** (Apple Silicon, Graviton, phones) | `x0` to `x7`, then stack | `x0` | frame pointer `x29`, link register `x30` |
| **RISC-V** | `a0` to `a7`, then stack | `a0` | return address in `ra` |
| **MIPS o32** (our code generation examples) | `$a0` to `$a3`, then stack | `$v0, $v1` | return address in `$ra` |

- Extra arguments go on the stack; so does anything bigger than the registers.
- **Caller-saved** registers may be overwritten by the callee; **callee-saved** ones must be restored.
- 32-bit x86 `cdecl` (arguments pushed right to left, caller cleans up) and `stdcall` are history; you meet them in old code and textbooks.

### **Why You Care in This Course**

Your compiler (final project) emits LLVM IR with plain `call i32 @f(i32 %x)`. **LLVM chooses the registers and builds the frame for each target** (x86-64, ARM64, RISC-V) following the platform ABI. You do not write prologues, but you need to understand what is being generated and why a call is not free.

---

## **The Stack in Action: Tracing Calls**

### **Trace the Program**

`main` calls `calculate`, which calls `multiply(5, 3)`. The state while `multiply` is running (x86-64, `-O0`, simplified; locals shown at the offsets gcc really used):

```
High address
┌──────────────────────────────────────┐
│ main frame                           │
│   return addr  -> OS startup code    │
│   saved rbp    -> OS startup code    │  <- main's rbp
│   final        (rbp-4)  ?            │
├──────────────────────────────────────┤
│ calculate frame                      │
│   return addr  -> inside main        │
│   saved rbp    -> main's rbp         │  <- calculate's rbp (0x20 below main's)
│   x            (rbp-4)  5            │
│   y            (rbp-8)  ?            │
├──────────────────────────────────────┤
│ multiply frame                       │
│   return addr  -> inside calculate   │
│   saved rbp    -> calculate's rbp    │  <- multiply's rbp (0x20 below calculate's)
│   result       (rbp-4)  15           │
└──────────────────────────────────────┘  <- rsp (top of the stack)
Low address
```

Remember the order inside every frame: **return address (highest), saved rbp, locals (lowest).** The arguments `a=5, b=3` arrived in registers `edi` and `esi`.

### **Calls and Returns Step by Step**

1. `main` executes `call calculate`: the CPU pushes the **return address**; `calculate` pushes `rbp` and sets `rbp = rsp`.
2. `calculate` stores `x = 5`, puts 5 and 3 into `edi` and `esi`, and executes `call multiply`.
3. `multiply` computes `result = 15`, puts it in `eax`, and **returns**: `pop rbp` restores `calculate`'s frame pointer, `ret` pops the return address.
4. `calculate` stores `y = 15`, returns `y + 1 = 16` in `eax`.
5. `main` stores `final = 16`.

The stack at the end of each step is exactly the diagram above with the top frame removed.

### **Do Not Trust the Picture: Print It**

```c
#include <stdio.h>
#include <stdint.h>
static void show(const char *fn, int *local) {
    uintptr_t *fp = __builtin_frame_address(1);   /* rbp of the calling function */
    printf("%-10s rbp=%p  saved rbp=%p  return addr=%p  local at rbp%+ld\n", fn,
           (void*)fp, (void*)fp[0], (void*)fp[1], (long)((char*)local - (char*)fp));
}
/* multiply, calculate and main call show("...", &local) as above */
```

Real output (`gcc -O0 -fno-omit-frame-pointer -fno-stack-protector`; addresses change on every run because of ASLR):

```
multiply   rbp=0x7ffe9f1b6cb0  saved rbp=0x7ffe9f1b6cd0  return addr=0x562374904216  local at rbp-4
calculate  rbp=0x7ffe9f1b6cd0  saved rbp=0x7ffe9f1b6cf0  return addr=0x562374904248  local at rbp-8
main       rbp=0x7ffe9f1b6cf0  saved rbp=0x7ffe9f1b6d90  return addr=0x7f1d46e2a1ca  local at rbp-4
```

- `multiply`'s saved `rbp` equals `calculate`'s `rbp`: the **control link** chain.
- Frames are 0x20 = 32 bytes apart: return address (8) + saved rbp (8) + locals (16, padded so `rsp` stays 16-byte aligned).
- Each frame sits **below** its caller's frame: the stack grows down.

---

## **Recursion, Stack Depth and Tail Calls**

### **Every Call Gets a Frame**

```ocaml
let rec fact n = if n <= 1 then 1 else n * fact (n - 1)     (* OCaml *)
```

```
fact 3          frames on the stack
  fact 2        [main] [fact 3] [fact 2] [fact 1]   <- deepest
    fact 1
```

The multiplication happens **after** the recursive call returns, so each frame must stay alive: depth n means n frames. The stack is finite (about 8 MB for the main thread on Linux):

| Language | What happens with a very deep recursion |
|----------|----------------------------------------|
| Python | `RecursionError: maximum recursion depth exceeded` (the interpreter's limit is 1000 by default) |
| C and compiled code of your project at `-O0` | Segmentation fault when the stack is full |
| Optimizing compiler | May turn some recursions into loops (see below) |

We checked the last two on the lecture's mini compiler: `count 100000` runs fine, `count 10000000` crashes at `-O0` and runs at `-O2` because LLVM rewrites it into a loop.

### **Tail Calls: A Call That Needs No Frame**

A call in **tail position** is the last thing a function does; nothing waits for its result.

```ocaml
let rec fact_acc n acc =            (* tail recursive *)
  if n <= 1 then acc else fact_acc (n - 1) (n * acc)
```

```
fact_acc 3 1  ->  fact_acc 2 3  ->  fact_acc 1 6  ->  6      (same frame reused: a jump)
```

- A **tail call** can be compiled as a **jump** that reuses the current frame. Constant stack space.
- **Functional languages depend on this** because they have no loops: iteration is recursion. OCaml and Scheme guarantee it; Rust and C compilers may do it as an optimization; **Python never does**.
- Your language has no loops either. Same story, one level lower: LLVM has a guaranteed form, `musttail call` (a bonus in your final project).

```llvm
rec:
  %n1 = sub i32 %n, 1
  %a1 = add i32 %acc, 1
  %r  = musttail call i32 @count(i32 %n1, i32 %a1)   ; must be followed by `ret %r`
  ret i32 %r
```

With this IR, a 100-million-deep recursion runs in constant stack at `-O0` (the backend emits `jmp count`). gcc at `-O2` also compiles `fact` and `fact_acc` as loops, but that is an optimization, not a promise.

---

## **Functions as Values: Nested Functions and Closures**

### **The Problem**

```ocaml
let make_adder n = fun x -> x + n      (* the inner function uses n from its parent *)
let add5 = make_adder 5                (* make_adder has returned: its frame is gone *)
let r = add5 10                        (* but n = 5 must still be available *)
```

A frame disappears when its function returns, but the inner function still needs `n`. Two classic solutions:

| Technique | Idea | When |
|-----------|------|------|
| **Static link (access link)** | Each frame stores a pointer to the frame of the lexically enclosing function | Nested functions that cannot outlive their parent (Pascal, Ada) |
| **Closure** | Function value = **code pointer + captured variables**, allocated on the **heap** | First-class functions that can escape (OCaml, JavaScript, Python, Rust, Java lambdas) |

### **A Closure in Memory**

```
add5 ─────► ┌────────────────────┐   heap
            │ code pointer       │──► code of the lambda
            │ captured n = 5     │
            └────────────────────┘
```

```c
typedef struct { int (*code)(void *env, int x); int n; } Closure;
int add_n(void *env, int x) { return x + ((Closure*)env)->n; }
Closure *c = malloc(sizeof *c);  c->code = add_n;  c->n = 5;
int r = c->code(c, 10);                                  /* 15 */
```

### **What Your Project Does**

Your language restricts this on purpose: an anonymous function **may not capture** local variables. Then a function value is just a **pointer to code** (`ptr @double` in LLVM IR): no environment, no heap, no closure record. A capturing lambda is rejected with a diagnostic, unless you take the closure-conversion bonus, which needs exactly the structure above.

---

## **Heap Management: Beyond the Stack**

### **When Stack Allocation Is Not Enough**

```java
class Node {
    int value;
    Node next;
    Node(int v) { value = v; next = null; }
}

Node head = new Node(1);        // the object is on the heap; `head` (a reference) is on the stack
head.next = new Node(2);
```

Data whose size or lifetime is not known when the function is compiled needs the **heap**: dynamic data structures, objects that outlive the creating function, closures that escape. (A JIT with escape analysis may avoid the allocation when an object provably does not escape; semantically it is still a heap object.)

### **Stack vs Heap Allocation in C**

```c
void stack_example(void) {
    int local_array[100];        // on the stack: size known at compile time
    local_array[0] = 42;
}                                // freed automatically when the function returns

void heap_example(int size) {
    int *a = malloc(size * sizeof(int));   // size known only at run time
    if (a != NULL) {
        a[0] = 42;
        free(a);                 // MUST be freed manually
    }
}
```

### **The Two Classic Bugs**

```c
void memory_leak(void) {
    int *data = malloc(1000);
    data[0] = 5;
    // forgot free(data): the 1000 bytes are lost until the program ends
}

void use_after_free(void) {
    int *ptr = malloc(sizeof(int));
    *ptr = 10;
    free(ptr);
    *ptr = 20;                   // writing to freed memory: undefined behavior
}
```

Tools find them: compile with `gcc -g -fsanitize=address` (AddressSanitizer), run, and read the report:

```
ERROR: AddressSanitizer: heap-use-after-free on address 0x502000000010
WRITE of size 4 at 0x502000000010 thread T0
    #0 main uaf.c:6
...
freed by thread T0 here:
    #0 free
    #1 main uaf.c:5
```

```
ERROR: LeakSanitizer: detected memory leaks
Direct leak of 1000 byte(s) in 1 object(s) allocated from:
    #0 malloc
    #1 main leak.c:2
```

Run your own C code with sanitizers during development (and your generated code too). These tools are also the right way to check C that an AI assistant wrote for you.

### **Heap Management Visualization**

```
Heap (grows up)               After free(block2), then malloc(300):
┌─────────────────┐           ┌─────────────────┐
│ block1: 100 B   │           │ block1: 100 B   │
│ free space      │           │ FREE: 200 B     │  <- too small for 300 B
│ block2: 200 B   │           │ block3: 50 B    │
│ block3: 50 B    │           └─────────────────┘
└─────────────────┘           external fragmentation: total free space is enough,
                              but no contiguous piece is
```

Production allocators (glibc `malloc`, jemalloc, tcmalloc, mimalloc) use size classes and per-thread arenas to keep this fast and to limit fragmentation.

---

## **Memory Management Strategies**

### **Who Frees the Memory?**

| Strategy | How it works | Pros | Cons | Used by |
|----------|--------------|------|------|---------|
| **Manual** | programmer calls `free` | full control, no runtime | leaks, use-after-free, double free | C, C++ (raw pointers) |
| **Ownership / RAII** | each value has one owner; memory is freed when the owner goes out of scope | no GC pauses, safe if the compiler checks it | steeper programming model | Rust (checked at compile time), C++ smart pointers |
| **Tracing garbage collection** | runtime finds unreachable objects (mark-sweep, copying, generational) | safe, productive | memory overhead, pauses (reduced by concurrent and generational collectors) | Java, Go, C#, OCaml, JavaScript |
| **Reference counting** | each object counts its references; freed at zero | predictable, immediate | cycles leak (need a cycle collector), counting cost | Swift (ARC), CPython (plus a cycle collector) |
| **Regions / arenas** | allocate many objects together, free the whole region at once | very fast, no per-object free | all-or-nothing lifetime | compilers (AST arenas!), game engines, request handlers |

> Your own compiler is a good arena candidate: AST nodes live exactly as long as one compilation. In C, allocating every node from one arena and freeing it once removes a whole class of bugs.

### **Stack vs Heap Trade-offs**

| Aspect | Stack | Heap |
|--------|-------|------|
| **Allocation speed** | Very fast (move the stack pointer) | Slower (search, bookkeeping) |
| **Deallocation** | Automatic on return | Manual, ownership scope, reference count, or GC |
| **Lifetime** | Until the function returns | As long as it is reachable or not freed |
| **Size** | Limited (about 8 MB on Linux, 1 MB on Windows by default) | Large (limited by the OS) |
| **Fragmentation** | None | Can occur |
| **Typical bugs** | Stack overflow (deep recursion, huge locals), buffer overruns that smash the return address | Leaks, use-after-free, double free, fragmentation |

Note that **neither** is bounds checked in C. Bounds checking is a language feature (Rust, Java, Go, Python), not a property of the stack or the heap.

---

## **Modern Memory Management**

### **Ownership: Compile-Time Memory Management (Rust)**

The Rust book states the rules: every value has an **owner**; there can be **one owner at a time**; when the owner goes out of scope the value is **dropped**.

```rust
fn process_data() {
    let data = vec![1, 2, 3];          // allocated on the heap, `data` owns it
    let processed = transform(data);   // ownership moves: `data` can no longer be used
    println!("{:?}", processed);
}                                      // `processed` goes out of scope: memory freed

fn transform(mut v: Vec<i32>) -> Vec<i32> {
    v.push(4);
    v                                  // ownership returned to the caller
}
```

Using `data` after the move is a **compile-time error**, so use-after-free cannot be written in safe Rust. This is the same idea as the type checker of week 8: a static analysis that rejects bad programs before they run.

### **Why Memory Safety Is a Policy Topic Now**

- A large share of serious security vulnerabilities in C and C++ code bases are memory-safety bugs (buffer overruns, use-after-free).
- In February 2024 the White House Office of the National Cyber Director published *Back to the Building Blocks: A Path Toward Secure and Measurable Software*, urging a shift toward **memory-safe languages**.
- Responses: Rust in new systems code, sanitizers and fuzzing for existing C, hardware features (pointer tagging, shadow stacks), and safer runtimes.

### **Other Trends**

- **Escape analysis** (Java, Go): objects that do not escape a function can live on the stack or in registers.
- **Generational and concurrent collectors** (JVM G1, ZGC, Shenandoah; the Go GC is concurrent, non-moving mark-sweep).
- **Pointer compression** (V8, JVM compressed object pointers): 32-bit references on 64-bit machines.
- **Better allocators** (jemalloc, mimalloc) and arenas.

---

## **How Types Influence the Runtime**

### **Semantic Analysis Feeds Layout**

```c
struct Point {
    int x, y;            /* 4 + 4 bytes */
    struct Point *next;  /* 8 bytes on a 64-bit machine */
};                       /* sizeof(struct Point) == 16 (checked with gcc on x86-64) */
```

```java
class Point { int x, y; Point next; }
// a Java object adds an object header (typically 8 to 16 bytes) and padding:
// on a typical 64-bit HotSpot JVM about 24 bytes (check with the JOL tool)
```

- The **type** tells the compiler the size, alignment and offsets of fields.
- Value types can live on the stack or in registers; reference types usually live on the heap.
- Type information also drives: **virtual method table** layout (next week), array **bounds-check elimination**, and which values the GC must scan.

---

## **Runtimes in the Real World**

### **Different Languages, Different Strategies**

**JavaScript (V8):**
- A pipeline of tiers: interpreter, baseline compilers, and an optimizing compiler (TurboFan)
- **Hidden classes** and **inline caches** for fast property access
- Generational garbage collector

**Java (JVM):**
- The JVM spec defines frames: *"Each frame has its own array of local variables, its own operand stack, and a reference to the run-time constant pool"* (JVMS 2.6), so the JVM is a **stack machine**
- Generational GC with low-pause collectors; JIT with profile-guided optimization; escape analysis

**Go:**
- Goroutine stacks start small and **grow dynamically** (copied to a bigger block when needed)
- Concurrent, non-moving mark-sweep GC
- Value types reduce heap pressure

**WebAssembly:**
- **Linear memory** model, sandboxed execution
- A stack machine whose call stack is managed by the engine (not part of the program-visible linear memory), so buffer overflows cannot overwrite return addresses
- Wasm 3.0 (September 2025) added garbage collection support and tail calls

---

## **AI Systems Are Runtimes Too**

### **The Same Ideas, A New Memory Problem**

- **KV cache paging.** LLM serving keeps the attention key/value cache of every request in GPU memory. The PagedAttention paper (vLLM, SOSP 2023) manages it **inspired by virtual memory and paging** in operating systems, to avoid fragmentation and waste. Same problem as the heap slides, same family of solutions.
- **Caching allocators.** Deep learning frameworks do not call the GPU allocator for every tensor; they keep a pool of blocks and reuse them (size classes, fragmentation again).
- **Static memory planning.** ML compilers such as XLA analyze buffers of the whole computation graph at compile time to allocate and schedule memory. This is what *static knowledge feeds the runtime* looks like when the whole program is a graph.
- **LLM-written systems code.** An assistant can write C that compiles and passes a few tests but leaks or overruns a buffer. Sanitizers and tests are the check; "it compiled" is not.

### **For Your Compiler**

The runtime of your final project is tiny: LLVM frames plus a C file (`runtime.c`) with the print functions. Think about what the extras would cost: closures need a heap, lists need allocation (is `malloc` without `free` acceptable, and when does it stop being acceptable?).

---

## **Project Implications**

### **What Your Final Project Gets for Free, and What It Does Not**

| Topic | In your compiler |
|-------|------------------|
| **Frames and registers** | LLVM allocates them; you emit `define`, `call`, `ret` |
| **Calling convention** | LLVM follows the platform ABI; `call i32 @f(i32 %x)` is enough |
| **Heap** | None in the baseline: no allocation, no closures, no lists |
| **Function values** | A `ptr` to a function: `call i32 %f(i32 %x)` |
| **Stack depth** | Deep recursion crashes the compiled program with a segmentation fault at `-O0`; `-O2` may turn it into a loop. Document it if you test it |
| **Division by zero** | Your runtime function prints a message to stderr and exits with a nonzero code |
| **`runtime.c`** | `print_int`, `print_bool`, and the division-by-zero handler |
| **Closure bonus** | Needs a heap-allocated environment, as in the closure slide |
| **Tail-call bonus** | `musttail` or loop lowering |

### **Three Paths, Same Runtime Contract**

- **C / Flex / Bison:** the compiler is a C program; use an arena for AST nodes.
- **Python / PLY:** the compiler runs on a VM that does all this for you; the compiled program still follows the LLVM/C runtime contract.
- **OCaml / Menhir:** garbage collected compiler, same output.

Whatever the path, the **output** is an `.ll` file linked with `runtime.c`: `clang prog.ll runtime.c -o prog`.

---

## **Practical Exercise: Stack Tracing**

### **Trace This Program (x86-64 style, arguments in registers)**

```c
int multiply(int a, int b) {
    int result = a * b;
    return result;
}

int calculate(void) {
    int x = 5;
    int y = multiply(x, 3);
    return y + 1;
}

int main(void) {
    int final = calculate();
    return final;
}
```

1. Draw the stack **when `multiply` is about to return**: three frames, each with return address, saved frame pointer and locals **in the right order**.
2. Which values are in each frame? Which registers carry `a` and `b` on Linux? On Windows?
3. What does the stack look like after `multiply` returns? After `calculate` returns?
4. What changes if the call were `multiply(1, 2, 3, 4, 5, 6, 7)` with seven arguments? (Where does the seventh go?)

### **Answers**

1. From the top (high addresses): `main` (`final = ?`), `calculate` (`x = 5`, `y = ?`), `multiply` (`result = 15`); inside each: return address, saved `rbp`, locals.
2. `a` in `edi`, `b` in `esi` (System V); `ecx` and `edx` (Windows).
3. After `multiply` returns, the `multiply` frame is gone and `y = 15`. After `calculate` returns, `final = 16`.
4. Six arguments travel in registers; the **seventh is pushed on the stack by the caller** (above the return address, at a positive offset from the callee's frame pointer).

### **Try It**

- Run the `frames.c` program from this lecture and check the 0x20 spacing.
- On https://godbolt.org compile `calculate` and `fact_acc` with `-O0` and `-O2`: where did the frame go?
- Write a recursive C function with depth 10 000 000 and run it. Then compile with `-O2`. What changed?

---

## **Key Takeaways**

1. **Memory is layered:** code, static data, heap, stack each serve a purpose.
2. **Activation records** hold the state of a call: arguments, return address, saved frame pointer, locals, in that order from high to low address.
3. **Calling conventions** are platform contracts (x86-64, ARM64, RISC-V, MIPS differ); LLVM applies them for you.
4. **Recursion costs stack space**; a **tail call** reuses the frame, which is why functional languages guarantee it.
5. **Function values:** a plain code pointer if nothing is captured; a heap-allocated **closure** if something is.
6. **Heap management** is a language design choice: manual, ownership, GC, reference counting, arenas.
7. **Modern runtimes** balance safety, speed and complexity, and AI systems rediscover the same techniques.

### **Next Lecture: Code Generation**

How do we turn the typed AST into LLVM IR (and, as a reference, MIPS), following exactly the rules of this lecture?

---

## **Further Reading and Resources**

**Official documents**
- Java Virtual Machine Specification, chapter 2 (frames): https://docs.oracle.com/javase/specs/jvms/se21/html/jvms-2.html
- Microsoft x64 calling convention: https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention
- Arm AAPCS64: https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
- RISC-V ELF psABI (calling convention): https://github.com/riscv-non-isa/riscv-elf-psabi-doc
- The Rust Book, chapter 4 (ownership): https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html
- A Guide to the Go Garbage Collector: https://go.dev/doc/gc-guide
- LLVM Language Reference (calls, `musttail`): https://llvm.org/docs/LangRef.html

**Books**
- Cooper and Torczon, *Engineering a Compiler* (the procedure abstraction)
- Appel, *Modern Compiler Implementation* (activation records)
- Aho et al., *Compilers: Principles, Techniques, and Tools* (Dragon Book), chapter 7
- Nystrom, *Crafting Interpreters* (free online): a bytecode VM, closures and garbage collection from scratch, https://craftinginterpreters.com
- Jones et al., *The Garbage Collection Handbook*

**Papers and reports**
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP 2023), https://arxiv.org/abs/2309.06180
- ONCD, *Back to the Building Blocks: A Path Toward Secure and Measurable Software* (February 2024)

**Hands-on**
- Compiler Explorer: https://godbolt.org
- `gcc -fsanitize=address`, `valgrind`
- Implement a simple stack machine; read JVM bytecode (`javap -c`) or WebAssembly text
