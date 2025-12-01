--

##  **Runtime Environments - Connecting Frontend to Backend**

### **Where We Are in the Compiler Pipeline**

```
Frontend:                    Backend:
Source Code                 Intermediate Code
    ↓                            ↓
Lexical Analysis    →    Runtime Environment Design
    ↓                            ↓
Parsing              →    Code Generation
    ↓                            ↓
Semantic Analysis    →    Optimization & Target Code
    ↓                            ↓
AST with Types       →    Executable Program
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
Low Address
┌─────────────────┐
│      Code       │  ← Program instructions (read-only)
├─────────────────┤
│   Static Data   │  ← Global variables, constants
├─────────────────┤
│       ↓         │
│      Heap       │  ← Dynamic memory allocation
│       ↑         │
├─────────────────┤
│       ↓         │
│     Stack       │  ← Function calls, local variables
└─────────────────┘
High Address
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

**Traditional C-style:**
```c
// Stack grows downward
┌─────────────────┐
│     Locals      │ ← Frame pointer
├─────────────────┤
│  Return Address │
├─────────────────┤
│   Parameters    │
├─────────────────┤
│  Control Link   │ ← Stack pointer
└─────────────────┘
```

---

**Modern Optimized (x86-64):**
```assembly
; Parameters in registers when possible
; Smaller frames, more register usage
calculate:
    push rbp
    mov rbp, rsp
    sub rsp, 16          ; Space for locals
    mov [rbp-4], edi     ; Save param1
    mov [rbp-8], esi     ; Save param2
    ; ... computation ...
    leave
    ret
```

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
**Stack Layout During bar(15, 3) Execution?**

---

**Stack Layout During bar(15, 3) Execution?**
```text
High Address
┌─────────────────┐
│    main frame   │
│   ───────────   │
│   result: ?     │ ← Frame pointer for main
├─────────────────┤ ← Stack grows downward
│    foo frame    │
│   ───────────   │
│   a: 10         │
│   b: 15         │
│   c: ?          │
│   ret addr: main+0x20  │
├─────────────────┤
│    bar frame    │ ← Current frame pointer
│   ───────────   │
│   x: 15         │ ← Parameters passed by value
│   y: 3          │
│   z: 45         │ ← Local variable
│   ret addr: foo+0x15   │
│   saved fp: →foo frame │ ← Control link
└─────────────────┘ ← Stack pointer
Low Address
```
---

### **Key Stack Operations:**
1. **Call `foo(10)`**: Push return address, push parameters
2. **Call `bar(15, 3)`**: Save current FP, push new frame
3. **Return from `bar`**: Store result, pop frame, restore FP
4. **Return from `foo`**: Same process

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
**Dynamic data structures are not supported in stack allocation**

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
| **Manual (malloc/free)** | Control, performance | Memory leaks, dangling pointers | C, C++ |
| **Garbage Collection** | Safety, productivity | Overhead, pauses | Java, Python, Go |
| **Automatic Reference Counting** | Predictable | Cyclic references | Swift, Objective-C |
| **Region-based** | Fast deallocation | Complex programming model | Rust (arenas) |

---

### **Heap vs Stack Trade-offs:**

| Aspect | Stack | Heap |
|--------|-------|------|
| **Allocation Speed** | Very fast (just move SP) | Slower (search + bookkeeping) |
| **Deallocation** | Automatic (function return) | Manual (free) or GC |
| **Lifetime** | Function scope | Until explicitly freed |
| **Size** | Limited (usually 1-8MB) | Large (limited by OS) |
| **Fragmentation** | None | Can occur |
| **Safety** | Automatic bounds checking | Manual management required |

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
- **Memory compression** (V8, .NET)

---

## **Project Implications - Runtime Choices**

### **How Your Compiler Design Affects Runtime**

**C/Flex/Bison Path:**
```c
// Manual stack management
void execute_function(Frame* frame) {
    // Explicit stack operations
    push_frame(frame);
    // ... execute ...
    pop_frame();
}
```
---

**Python/PLY Path:**
```python
# Interpreter-based runtime
class VirtualMachine:
    def __init__(self):
        self.call_stack = []
        self.heap = {}
    
    def call_function(self, func, args):
        frame = Frame(func, args)
        self.call_stack.append(frame)
        result = self.execute_frame(frame)
        self.call_stack.pop()
        return result
```
---

**OCaml/LLVM Path:**
```ocaml
(* LLVM handles runtime details *)
let compile_function name params body =
  let fn = define_function name (function_type i32_type params) the_module in
  let builder = builder_at_end context (entry_block fn) in
  (* LLVM generates optimal stack/heap code *)
```

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
- **Deterministic performance**

---

## **Connecting Back to Semantic Analysis**

### **How Types Influence Runtime**

```java
// Semantic analysis tells us:
class Point {
    int x, y;        // Each Point: 8 bytes
    Point next;      // Reference: 4/8 bytes
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


#### **Step 1: main() calls calculate()**
```
Stack after main() calls calculate():
┌─────────────────┐
│    main frame   │
│   ───────────   │
│   final: ?      │
│   ret addr: OS  │
│   saved fp: OS  │
├─────────────────┤
│ calculate frame │ ← Current FP
│   ───────────   │
│   x: ?          │
│   y: ?          │
│   ret addr: main+0x10 │
│   saved fp: →main│
└─────────────────┘ ← SP
```

---

#### **Step 2: calculate() sets x = 5 and calls multiply(5, 3)**
```
Stack during multiply(5, 3) execution:
┌─────────────────┐
│    main frame   │
│   ───────────   │
│   final: ?      │
├─────────────────┤
│ calculate frame │
│   ───────────   │
│   x: 5          │
│   y: ?          │
│   ret addr: main+0x10│
│   saved fp: →main│
├─────────────────┤
│ multiply frame  │ ← Current FP
│   ───────────   │
│   a: 5          │ ← Parameter (value = 5)
│   b: 3          │ ← Parameter (value = 3)
│   result: ?     │ ← Local variable
│   ret addr: calculate+0x15│
│   saved fp: →calculate│
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
│   ───────────   │
│   final: ?      │
├─────────────────┤
│ calculate frame │ ← Current FP
│   ───────────   │
│   x: 5          │
│   y: 15         │ ← Return value stored here
│   ret addr: main+0x10│
│   saved fp: →main│
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
│   ───────────   │
│   final: 16     │ ← Return value stored here
│   ret addr: OS  │
│   saved fp: OS  │
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
Stack grows downward (high → low addresses):

0xFFFFF000: main frame starts
   final @ 0xFFFFF000
   ret addr @ 0xFFFFF004
   saved fp @ 0xFFFFF008

0xFFFFEF00: calculate frame starts (when called)
   x @ 0xFFFFEF00
   y @ 0xFFFFEF04
   ret addr @ 0xFFFFEF08
   saved fp @ 0xFFFFEF0C (points to 0xFFFFF000)

0xFFFFEE00: multiply frame starts (when called)
   a @ 0xFFFFEE00
   b @ 0xFFFFEE04
   result @ 0xFFFFEE08
   ret addr @ 0xFFFFEE0C
   saved fp @ 0xFFFFEE10 (points to 0xFFFFEF00)
```

---

### **Key Observations:**
1. **Parameter passing**: By value, so copies of `x` and `3` are made in `multiply`'s frame
2. **Return values**: Typically passed in registers, not on stack
3. **Frame pointers**: Each frame points to previous frame for easy unwinding
4. **Stack cleanup**: Caller or callee responsibility depends on calling convention

---

### **Common Calling Conventions:**
- **cdecl (C)**: Caller cleans stack, parameters pushed right-to-left
- **stdcall (Windows API)**: Callee cleans stack
- **fastcall**: First few parameters in registers

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

## **Key Takeaways**

### **What Every Compiler Designer Needs to Know**

1. **Memory is layered**: Code, static, heap, stack each serve different purposes
2. **Activation records** manage function execution state
3. **Stack discipline** enables function calls and returns
4. **Heap management** strategies vary by language philosophy
5. **Modern runtimes** balance safety, performance, and complexity

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

