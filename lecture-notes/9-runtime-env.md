##  **Runtime Environments - Connecting Frontend to Backend**

### **Where We Are in the Compiler Pipeline**

```
Source -> Lexer -> Parser -> AST -> Semantic Analysis -> IR -> Code Generation -> Executable
                                    (types, scopes)               ^
                                                                  |
                      Runtime environment: the contract the generated code runs under
```

---


### **The Big Question:**
**"Now that we understand the program's structure and types, how do we actually make it run?"**

---

## **From Static Analysis to Dynamic Execution**

### **Static vs Dynamic World**

| Static (Compile-time) | Dynamic (Runtime) |
|---------------------|------------------|
| AST Structure | Activation Records |
| Type Information | Memory Layout |
| Symbol Tables | Stack Frames |
| Scope Rules | Procedure Calls |
| **Known at compile time** | **Determined during execution** |

---

### **Key Insight:**
The runtime environment bridges our **static understanding** of the program with **dynamic execution**.

---

## **Memory Layout - The Program's Home**

### **Traditional Memory Organization**

```
High Address
┌─────────────────┐
│     Stack       │  ← Function calls, local variables
│       ↓         │     (grows toward lower addresses)
│                 │
│       ↑         │
│      Heap       │  ← Dynamic memory allocation (grows up)
├─────────────────┤
│   Static Data   │  ← Global variables, constants
├─────────────────┤
│      Code       │  ← Program instructions (read-only)
└─────────────────┘
Low Address
```

---


### **Modern Reality:**
- Virtual memory makes this more flexible
- Multiple threads = multiple stacks
- Security considerations (stack guards, ASLR)

---

## **Activation Records - The Heart of Execution**

### **What Happens When You Call a Function?**

```c
int calculate(int x, int y) {
    int result = x * y;
    return result + 10;
}

int main() {
    int a = calculate(5, 3);
    return a;
}
```

---

### **Activation Record (Stack Frame) Contents:**

| Component | Purpose | Example |
|-----------|---------|---------|
| **Return Address** | Where to continue after function | `main+0x15` |
| **Parameters** | Function arguments | `x=5, y=3` |
| **Local Variables** | Function-scoped data | `result=15` |
| **Temporary Storage** | Intermediate calculations | `(none)` |
| **Control Link** | Caller's frame pointer | `→ main's frame` |

---


## **Modern Activation Record Design**

### **Traditional vs Modern Approaches**

**Traditional C-style** (high address at the top, the stack grows down):
```
┌─────────────────┐
│   Parameters    │ ← pushed by the caller (if not in registers)
├─────────────────┤
│  Return Address │ ← pushed by the `call` instruction
├─────────────────┤
│    Saved FP     │ ← Frame pointer points here (control link)
├─────────────────┤
│     Locals      │ ← FP-4, FP-8, ...
├─────────────────┤
│   Temporaries   │ ← Stack pointer
└─────────────────┘
```
Order from high to low address: **parameters, return address, saved FP, locals, temporaries.**

---

**x86-64 (System V ABI: Linux, macOS), `gcc -O0`:**
```assembly
; First arguments arrive in registers (edi, esi), then are spilled to the frame
calculate:
    push rbp
    mov rbp, rsp
    mov [rbp-20], edi    ; Save param1
    mov [rbp-24], esi    ; Save param2
    ; ... computation: result at [rbp-4] ...
    pop rbp
    ret
```

---

**Same function with `gcc -O2`: no frame at all**
```assembly
calculate:
    imul edi, esi        ; x * y
    lea  eax, 10[rdi]    ; + 10, result in eax
    ret
```
- A frame is an implementation choice, not a law: with optimization everything stays in registers.
- Windows x64 passes arguments in `rcx, rdx, r8, r9` instead; ARM64 in `x0-x7`.
- Recent Fedora and Ubuntu 24.04 enable frame pointers by default again, so profilers can walk the stack.
- Try it: https://godbolt.org

---

## **The Stack in Action - Live Example**

### **Tracing Execution**

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(3)
```
```
factorial(3)
└─ factorial(2)
   └─ factorial(1)       <- the activation tree: each call is a node
```
*An activation tree shows the way control enters and leaves activations.*

---

### **Stack Growth:**
```
Initial: [main]
Call factorial(3): [main, factorial(3)]
Call factorial(2): [main, factorial(3), factorial(2)]
Call factorial(1): [main, factorial(3), factorial(2), factorial(1)]
Return 1: [main, factorial(3), factorial(2)]  # 2 * 1 = 2
Return 2: [main, factorial(3)]               # 3 * 2 = 6
Return 6: [main]                             # result = 6
```

---

**Explicit Stack Activation Record Example**
```C
// Example program showing explicit stack frames
int bar(int x, int y) {
    int z = x * y;      // Local variable
    return z + 1;
}
int foo(int a) {
    int b = a + 5;
    int c = bar(b, 3);  // Function call
    return c * 2;
}
int main() {
    int result = foo(10);
    return result;
}

```

---

**Stack Layout During bar(15, 3) Execution**
```text
High Address
┌──────────────────────────┐
│ main frame               │
│   ret addr: OS           │
│   saved fp: OS           │ ← main's frame pointer
│   result: ?              │
├──────────────────────────┤ ← Stack grows downward
│ foo frame                │
│   ret addr: main+0x20    │
│   saved fp: → main frame │ ← foo's frame pointer
│   a: 10                  │
│   b: 15                  │
│   c: ?                   │
├──────────────────────────┤
│ bar frame                │
│   ret addr: foo+0x15     │
│   saved fp: → foo frame  │ ← bar's frame pointer (current), control link
│   x: 15                  │ ← parameters (spilled from registers here)
│   y: 3                   │
│   z: 45                  │ ← local variable
└──────────────────────────┘ ← Stack pointer
Low Address
```
Order inside every frame: **return address, saved frame pointer, then locals.**

---

### **Key Stack Operations:**
1. **Call `foo(10)`**: caller puts the arguments in place (registers or stack), `call` pushes the return address
2. **Callee prologue**: push the old FP, set FP = SP, reserve space for locals
3. **Return from `bar`**: result in the return register, restore FP, pop the frame, jump to the return address
4. **Return from `foo`**: same process

---

## **Recursion and Tail Calls**

```python
# Regular recursion — O(n) stack space
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # work happens AFTER the call returns

# Tail recursion — O(1) stack space (with TCO)
def factorial_tail(n, acc=1):
    if n <= 1:
        return acc
    return factorial_tail(n - 1, n * acc)  # call is the LAST thing done   

```
---

```python
#Naive: O(2^n) time, O(n) stack
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)  # work AFTER the call (the +)   

# Tail recursion-O(1) stack, O(n) time
def fib_tail(n, a=0, b=1):
    if n == 0:
        return a
    return fib_tail(n - 1, b, a + b)   
```

---

### **Every Call Gets a Frame**

```ocaml
let rec fact n = if n <= 1 then 1 else n * fact (n - 1)        (* O(n) frames *)
let rec fact_acc n acc =                                       (* tail call *)
  if n <= 1 then acc else fact_acc (n - 1) (n * acc)           (* O(1) frames *)
```

- `fact`: the `*` waits for the call to return → each frame must stay alive.
- `fact_acc`: the call **is** the last thing done → compiler replaces it with a jump.

The stack is finite. Without TCO, deep recursion overflows it (Python: `RecursionError` at ~1000; C/LLVM: segfault).

**Language support:** OCaml and Scheme guarantee TCO; gcc does it at `-O2`; Python never does.  LLVM has a guaranteed form: `musttail call`.

---

## **Functions as Values: Closures**

```ocaml
let make_adder n = fun x -> x + n      (* inner function uses n from the parent *)
let add5 = make_adder 5                (* the frame of make_adder is gone ... *)
let r = add5 10                        (* ... but n = 5 is still needed *)
```

```
add5 ───► ┌─────────────────┐  heap
          │ code pointer    │──► code of the lambda
          │ captured n = 5  │
          └─────────────────┘
```

- A **closure** = code pointer + captured variables, allocated on the **heap** (so it can outlive the frame).
- Nested functions that cannot escape can use a **static link** (a pointer to the parent's frame) instead.
- **Final project:** lambdas may **not capture** local variables, so a function value is just a pointer to code (`ptr @double` in LLVM IR): no heap, no closure. Capturing lambdas are the closure-conversion bonus.

---

## **Heap Management - Beyond the Stack**

### **When Stack Allocation Isn't Enough**

```java
class Node {
    int value;
    Node next;
    
    Node(int v) { 
        value = v; 
        next = null;
    }
}

// These must live on the heap!
Node head = new Node(1);
head.next = new Node(2);
```
**Data whose size or lifetime is not known at compile time cannot live in a stack frame**

---

### **Dynamic vs Stack Allocation**

```c
// Stack allocation (compile-time known size)
void stack_example() {
    int local_array[100];     // On stack - auto freed
    local_array[0] = 42;
    // Array automatically freed when function returns
}

// Heap allocation (runtime determined size)
void heap_example(int size) {
    // Allocate on heap - persists until explicitly freed
    int* dynamic_array = (int*)malloc(size * sizeof(int));
    
    if (dynamic_array != NULL) {
        dynamic_array[0] = 42;
        // ... use the array ...
        free(dynamic_array);  // MUST free manually!
    }
}

// Common bug: memory leak
void memory_leak() {
    int* data = malloc(1000);
    data[0] = 5;
    // OOPS! Forgot to free(data) - memory leak!
    // The 1000 bytes are lost until program ends
}

// Dangling pointer bug
void dangling_pointer() {
    int* ptr = malloc(sizeof(int));
    *ptr = 10;
    free(ptr);        // Memory freed
    *ptr = 20;        // ERROR: writing to freed memory!
}
```

---

### **Heap Management Visualization:**

```
Memory Layout:
┌─────────────────┐
│      Code       │
├─────────────────┤
│   Static Data   │
├─────────────────┤
│                 │
│      Heap       │ ← Dynamic allocations grow upward
│   ───────────   │
│   [block1: 100B]│ ← malloc(100)
│   [free space]  │
│   [block2: 200B]│ ← malloc(200)
│   [block3: 50B] │ ← malloc(50)
│                 │
├─────────────────┤
│     Stack       │ ← Function frames grow downward
│   ───────────   │
│   foo() frame   │
│   main() frame  │
└─────────────────┘
```

---

```
Heap Fragmentation Example:
After free(block2) and malloc(300):
┌─────────────────┐
│   [block1: 100B]│
│   [FREE: 200B]  │ ← Cannot use for 300B allocation!
│   [block3: 50B] │
└─────────────────┘
→ Need to search for larger contiguous space
```

---

### **Heap Allocation Strategies:**

| Strategy | Pros | Cons | Used By |
|----------|------|------|---------|
| **Manual (malloc/free)** | Control, performance | Memory leaks, dangling pointers | C, C++ (raw pointers) |
| **Ownership / RAII** | No GC pauses, checked at compile time in Rust | Learning curve | Rust, C++ smart pointers |
| **Tracing garbage collection** | Safety, productivity | Overhead, pauses | Java, Go, C#, OCaml, JavaScript |
| **Reference counting** | Predictable, immediate | Cyclic references, counting cost | Swift (ARC), CPython (plus a cycle collector) |
| **Region / arena** | Fast, free everything at once | All-or-nothing lifetime | Compilers (AST nodes!), game engines |

Tip for your own compiler in C: allocate AST nodes from one arena and free it once.

---

### **Heap vs Stack Trade-offs:**

| Aspect | Stack | Heap |
|--------|-------|------|
| **Allocation Speed** | Very fast (just move SP) | Slower (search + bookkeeping) |
| **Deallocation** | Automatic (function return) | Manual (free) or GC |
| **Lifetime** | Function scope | Until explicitly freed |
| **Size** | Limited (usually 1-8MB) | Large (limited by OS) |
| **Fragmentation** | None | Can occur |
| **Typical bugs** | Stack overflow, buffer overruns | Leaks, use-after-free, fragmentation |

---

## **Modern Memory Management Trends**

### **Zero-Cost Abstractions & Safety**

**Rust's Ownership System:**
```rust
fn process_data() {
    let data = vec![1, 2, 3];  // Allocated on heap
    let processed = transform(data);  // Ownership transferred
    // data is no longer accessible here
    println!("{:?}", processed);
}  // Memory automatically freed

fn transform(mut vec: Vec<i32>) -> Vec<i32> {
    vec.push(4);
    vec  // Ownership returned
}
```

### **Key Innovations:**
- **Compile-time memory management** (Rust)
- **Escape analysis** (Java, Go)
- **Generational garbage collectors** (most modern VMs)
- **Pointer compression** (V8, JVM compressed oops)

---

## **Project Implications - Runtime Choices**

### **What Your Final Project Gets for Free, and What It Does Not**

| Topic | In your compiler |
|-------|------------------|
| **Frames, registers, calling convention** | LLVM does it for the target: you emit `define`, `call`, `ret` |
| **Heap** | None in the baseline: no allocation, no closures, no lists |
| **Function values** | A `ptr` to a function: `call i32 %f(i32 %x)` |
| **Deep recursion** | Compiled program may crash with a segmentation fault at `-O0` (stack is finite); `-O2` may turn it into a loop |
| **Division by zero** | `runtime.c` function prints to stderr and exits with a nonzero code |
| **Bonuses** | Closures need a heap-allocated environment; tail calls need `musttail` or loop lowering |

---

### **All Three Paths Share One Runtime Contract**

- **C / Flex / Bison, Python / PLY, OCaml / Menhir:** your compiler emits an `.ll` file, linked with `runtime.c`:

```bash
clang prog.ll runtime.c -o prog
```

- `runtime.c` holds `print_int`, `print_bool` and the division-by-zero handler. Everything else (stack frames, registers) is generated by LLVM.
- Next lecture: how to generate that IR. A tested mini example is in `10-code-generation-example/`.

---

## **Real-World Runtime Examples**

### **Different Languages, Different Strategies**

**JavaScript (V8 Engine):**
- **Hidden classes** for fast property access
- **Inline caching** for method calls
- **Turbofan compiler** with escape analysis

---

**Java (JVM):**
- **Precise garbage collection** with generations
- **JIT compilation** with profile-guided optimization
- **Escape analysis** for stack allocation

---

**Go:**
- **Goroutine stacks** that grow dynamically
- **Concurrent garbage collection**
- **Value types** to reduce heap pressure
---

**WebAssembly:**
- **Linear memory** model
- **Sandboxed execution**
- **Near-native, predictable performance**
- Call stack is not in linear memory (return addresses cannot be overwritten by a buffer overflow)

---

## **Connecting Back to Semantic Analysis**

### **How Types Influence Runtime**

```java
// Semantic analysis tells us:
class Point {
    int x, y;        // 8 bytes of data
    Point next;      // Reference: 4/8 bytes, plus an object header (about 24 bytes in total on a 64-bit JVM)
}

// Runtime implications:
Point p = new Point(); 
// Heap allocation: object header + 8 bytes + reference
// Stack: reference variable (4/8 bytes)
```
---

### **Type-Driven Optimizations:**
- **Stack allocation** for value types
- **Memory layout** optimization
- **Virtual method table** design
- **Array bounds check** elimination

---

## **Practical Exercise - Stack Tracing**

### **Trace This Program's Execution**

```c
int multiply(int a, int b) {
    int result = a * b;
    return result;
}

int calculate() {
    int x = 5;
    int y = multiply(x, 3);
    return y + 1;
}

int main() {
    int final = calculate();
    return final;
}
```

---

### **Draw the Stack Frames:**
1. **main()** calls **calculate()**
2. **calculate()** calls **multiply(5, 3)**
3. **multiply()** returns to **calculate()**
4. **calculate()** returns to **main()**

**Question:** What's in each activation record?

---

### **Execution Trace & Stack Frames:**

(Order inside every frame: return address, saved fp, locals.)

#### **Step 1: main() calls calculate()**
```
Stack after main() calls calculate():
┌─────────────────┐
│    main frame   │
│   ret addr: OS  │
│   saved fp: OS  │
│   final: ?      │
├─────────────────┤
│ calculate frame │
│   ret addr: main+0x10 │
│   saved fp: →main│ ← Current FP
│   x: ?          │
│   y: ?          │
└─────────────────┘ ← SP
```

---

#### **Step 2: calculate() sets x = 5 and calls multiply(5, 3)**
```
Stack during multiply(5, 3) execution:
┌─────────────────┐
│    main frame   │
│   ret addr: OS  │
│   saved fp: OS  │
│   final: ?      │
├─────────────────┤
│ calculate frame │
│   ret addr: main+0x10│
│   saved fp: →main│
│   x: 5          │
│   y: ?          │
├─────────────────┤
│ multiply frame  │
│   ret addr: calculate+0x15│
│   saved fp: →calculate│ ← Current FP
│   a: 5          │ ← Parameter (value = 5)
│   b: 3          │ ← Parameter (value = 3)
│   result: ?     │ ← Local variable
└─────────────────┘ ← SP
```

---

#### **Step 3: multiply() computes result = 15 and returns**
- **multiply frame**: `result = 5 * 3 = 15`
- Return value 15 is placed in return value register (e.g., EAX in x86)
- Frame popped, SP moves up, FP restored to calculate frame

---

#### **Step 4: Back in calculate(), y = 15, then returns y + 1 = 16**
```
Stack after multiply returns:
┌─────────────────┐
│    main frame   │
│   ret addr: OS  │
│   saved fp: OS  │
│   final: ?      │
├─────────────────┤
│ calculate frame │
│   ret addr: main+0x10│
│   saved fp: →main│ ← Current FP
│   x: 5          │
│   y: 15         │ ← Return value stored here
└─────────────────┘ ← SP
```
- calculate() computes: `y + 1 = 15 + 1 = 16`
- Return value 16 placed in return register
- Frame popped, SP moves up, FP restored to main frame

---

#### **Step 5: Back in main(), final = 16, then returns**
```
Stack after calculate returns:
┌─────────────────┐
│    main frame   │ ← Current FP
│   ret addr: OS  │
│   saved fp: OS  │
│   final: 16     │ ← Return value stored here
└─────────────────┘ ← SP
```

---

### **Complete Activation Record Contents:**

| Function | Frame Contents |
|----------|----------------|
| **multiply** | - Parameters: `a=5`, `b=3`<br>- Local: `result=15`<br>- Return address: `calculate+0x15`<br>- Control link: →calculate frame |
| **calculate** | - Locals: `x=5`, `y=15`<br>- Return address: `main+0x10`<br>- Control link: →main frame |
| **main** | - Local: `final=16`<br>- Return address: OS (caller)<br>- Control link: OS frame |

---

### **Memory Addresses (Hypothetical 32-bit):**
```
Stack grows downward (high → low addresses). Frames are contiguous:
return address, saved fp, then locals at lower addresses.

0xFFFFF00C  main:      ret addr  (OS)
0xFFFFF008             saved fp  (OS)               ← main's FP
0xFFFFF004             final
0xFFFFF000  calculate: ret addr  (main+0x10)
0xFFFFEFFC             saved fp  = 0xFFFFF008       ← calculate's FP
0xFFFFEFF8             x
0xFFFFEFF4             y
0xFFFFEFF0  multiply:  ret addr  (calculate+0x15)
0xFFFFEFEC             saved fp  = 0xFFFFEFFC       ← multiply's FP
0xFFFFEFE8             a
0xFFFFEFE4             b
0xFFFFEFE0             result
```
The saved fp of each frame points to the frame pointer of its caller: the **control link chain**.

---

### **Key Observations:**
1. **Parameter passing**: By value, so copies of `x` and `3` are made in `multiply`'s frame
2. **Return values**: Typically passed in registers, not on stack
3. **Frame pointers**: Each frame points to previous frame for easy unwinding
4. **Stack cleanup**: Caller or callee responsibility depends on calling convention

---

### **Common Calling Conventions:**

| Platform | Integer arguments | Return |
|----------|-------------------|--------|
| **x86-64 System V** (Linux, macOS) | `rdi, rsi, rdx, rcx, r8, r9`, then stack | `rax` |
| **x86-64 Windows** | `rcx, rdx, r8, r9`, then stack (caller reserves shadow space) | `rax` |
| **ARM64** (Apple Silicon, phones) | `x0` to `x7`, then stack | `x0` |
| **RISC-V** | `a0` to `a7`, then stack | `a0` |
| **MIPS o32** (our examples) | `$a0` to `$a3`, then stack | `$v0, $v1` |

- **Caller-saved** registers may be overwritten by the callee; **callee-saved** ones must be restored.
- 32-bit x86 `cdecl` (arguments pushed right to left, caller cleans up) and `stdcall` are historical.
- Your LLVM-based compiler never writes these rules: LLVM applies the ABI of the target.

---

## **Modern Challenges & Solutions**

### **Contemporary Runtime Issues**

**1. Concurrency:**
```go
func processConcurrently() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            processItem(id)  // Each goroutine needs its own stack
        }(i)
    }
    wg.Wait()
}
```
---

**2. Memory Safety:**
- **Bounds checking** (Rust, Go)
- **Use-after-free detection** (AddressSanitizer)
- **Stack overflow protection** (stack guards)

**3. Performance:**
- **JIT compilation** with adaptive optimization
- **Ahead-of-time compilation** for predictable performance
- **Profile-guided optimization**

---

## **AI Systems Are Runtimes Too**

- **KV cache paging:** LLM serving manages the attention cache in blocks, inspired by operating-system virtual memory (PagedAttention / vLLM, SOSP 2023).
- **Caching allocators and static memory planning:** deep learning frameworks reuse GPU memory blocks; ML compilers such as XLA analyze buffers at compile time.
- **Memory safety is a policy topic:** the White House ONCD report (Feb 2024) urged memory-safe languages. Run AI-written C with sanitizers: `gcc -g -fsanitize=address`.

---

## **Key Takeaways**

### **What Every Compiler Designer Needs to Know**

1. **Memory is layered**: Code, static, heap, stack each serve different purposes
2. **Activation records** manage function execution state
3. **Stack discipline** enables function calls and returns
4. **Heap management** strategies vary by language philosophy
5. **Modern runtimes** balance safety, performance, and complexity
6. **Recursion costs stack space**; a tail call reuses the frame (why functional languages guarantee it)
7. **Closures** need a heap-allocated environment; **no capture** means a plain code pointer

### **Next Lecture Preview:**
**Code Generation** - How we translate our understood program into actual machine instructions that use this runtime environment efficiently!

---

## **Further Reading & Resources**

### **Deep Dives:**

**Academic:**
- "The Garbage Collection Handbook" by Jones et al.
- "Compilers: Principles, Techniques, and Tools" (Dragon Book) Chapter 7

---

**Practical:**
- **V8 Blog**: https://v8.dev/blog
- **Java HotSpot VM Internals**
- **Rustonomicon**: https://doc.rust-lang.org/nomicon/

---

**Hands-on Practices:**
- Implement a simple stack machine
- Study JVM bytecode or WebAssembly
- Experiment with different garbage collection algorithms
- Compare `gcc -O0` and `-O2` on https://godbolt.org
- *Crafting Interpreters* (free): https://craftinginterpreters.com
- JVM spec, frames: https://docs.oracle.com/javase/specs/jvms/se21/html/jvms-2.html
- Rust Book, ownership: https://doc.rust-lang.org/book/ch04-01-what-is-ownership.html
- LLVM LangRef: https://llvm.org/docs/LangRef.html

