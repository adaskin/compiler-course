- [Introduction to AI Compilers: From Tensors to Specialized Hardware](#introduction-to-ai-compilers-from-tensors-to-specialized-hardware)
  - [The AI Compilation Pipeline](#the-ai-compilation-pipeline)
  - [Why Traditional Compilers Aren't Enough](#why-traditional-compilers-arent-enough)
  - [Why We Need AI Compilers](#why-we-need-ai-compilers)
  - [The Mathematical Foundation of AI models](#the-mathematical-foundation-of-ai-models)
    - [Higher-Dimensional Matrices](#higher-dimensional-matrices)
    - [What Are Tensors? The Basic Data Structure of AI](#what-are-tensors-the-basic-data-structure-of-ai)
    - [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
  - [The Hardware Evolution: From Scalar to Tensor Processing](#the-hardware-evolution-from-scalar-to-tensor-processing)
    - [1. Scalar Processors: The General-Purpose Workhorse](#1-scalar-processors-the-general-purpose-workhorse)
    - [2. Vector Processors: Doing Many Things at Once](#2-vector-processors-doing-many-things-at-once)
    - [3. Tensor Processors: Specialized for AI](#3-tensor-processors-specialized-for-ai)
    - [Real Hardware Examples](#real-hardware-examples)
    - [Hardware Specialization Spectrum](#hardware-specialization-spectrum)
    - [Why This Matters for AI Compilers](#why-this-matters-for-ai-compilers)
  - [The AI Compiler: Bridging Models and Hardware](#the-ai-compiler-bridging-models-and-hardware)
    - [The AI Compilation Pipeline](#the-ai-compilation-pipeline-1)
    - [AI Compiler Ecosystem in 2025](#ai-compiler-ecosystem-in-2025)
    - [Real-World Impact: Case Study](#real-world-impact-case-study)
    - [Hands-On: From Tensor Operations to Compiled Code](#hands-on-from-tensor-operations-to-compiled-code)
    - [AI Framework Comparison: PyTorch vs JAX vs TensorFlow](#ai-framework-comparison-pytorch-vs-jax-vs-tensorflow)
    - [JAX Example: Functional and Composable](#jax-example-functional-and-composable)
    - [From Framework Code to Optimized Hardware](#from-framework-code-to-optimized-hardware)
  - [Why This Matters for You](#why-this-matters-for-you)
  - [Further Learning Path](#further-learning-path)
    - [From Here to AI Compiler Expert](#from-here-to-ai-compiler-expert)

# Introduction to AI Compilers: From Tensors to Specialized Hardware

---

## The AI Compilation Pipeline
- In traditional programming, we write code that describes how to compute something. 
- In AI, we define what to compute as:

```
      [Input Image]
          ↓
    [Convolution]
          ↓
   [Batch Normalization]
          ↓
      [ReLU]
          ↓
    [Max Pooling]
          ↓
     [Output]
```

---

**as a computational graph**
```
AI Framework (PyTorch/TensorFlow)
         ↓
   Graph Extraction
         ↓
Computational Graph (ONNX/TorchScript)
         ↓
  Graph Optimizations
         ↓
Hardware-Specific Lowering  
         ↓
   Kernel Generation
         ↓
   Accelerator Code
```
 *This graph structure—not text code—is what AI compilers work with.*

---

---
## Why Traditional Compilers Aren't Enough

**Traditional Compiler Flow:**
```
C/Java/Python Code → Lexing/Parsing → AST → Optimizations → Machine Code
```

**Problems for AI Models:**
1. **Different input**: AI models are computational graphs, not text
2. **Different operations**: Tensor operations vs scalar operations
3. **Different optimization goals**: Numerical stability, memory hierarchy, parallel execution
4. **Diverse targets**: Need to run on GPUs, TPUs, phones, edge devices—all with different capabilities
   
---

This diversity in hardware targets means traditional compilers, which are designed for CPUs, can't automatically adapt.
**Example Challenge:** A single AI model might need to run on:
- An NVIDIA GPU in a data center
- An iPhone's Neural Engine
- A Raspberry Pi with limited memory
- A Google TPU in the cloud
 For example, a single AI model might need to run on  

Traditional compilers can't handle this diversity automatically. 

---

## Why We Need AI Compilers
**Before AI Compilers**
**Traditional AI Deployment Workflow:**
```python
# Researcher's code (Python/PyTorch)
model = MyNeuralNetwork()
```
**Engineer's nightmare (months of work):**
1. Convert PyTorch to TensorFlow
2. Hand-write CUDA kernels for NVIDIA GPUs
3. Write Metal shaders for Apple devices
4. Optimize ARM NEON code for Android
5. Create separate branches for each hardware
6. Performance tuning for each platform
   
---


**Typical Results:**
- **Inference latency**: 100ms
- **Power consumption**: 10W
- **Model size**: 500MB (FP32)
- **Deployment time**: 6+ months
- **Hardware support**: Limited to 1-2 platforms

---

**With Modern AI Compilers: Write Once, Run Anywhere**

**Today's Workflow with AI Compilers:**
```python
# One implementation (Python/PyTorch/JAX)
model = MyNeuralNetwork()

# Single compilation step
compiled_model = compile_for_target(
    model=model,
    target="auto",  # Auto-detects hardware
    optimization_level=3
)
# Runs everywhere with optimal performance
results = compiled_model(input_data)
```

---

**Achievable Results:**
- **Inference latency**: 10ms (10× faster)
- **Power consumption**: 2W (5× more efficient)
- **Model size**: 125MB (4× smaller with quantization)
- **Deployment time**: Hours instead of months
- **Hardware support**: Cloud GPUs, mobile phones, edge devices

---

## The Mathematical Foundation of AI models

**Matrix Operations (What You Know):**
- A matrix is a 2D grid of numbers: `A[rows][columns]`
- Basic operation: Matrix multiplication

```
C = A × B
where C[i][j] = Σ A[i][k] × B[k][j]
```

**In Neural Networks:**
- A single image classification might require thousands of matrix multiplications
- Each layer transforms data through matrix operations
- Example: A simple fully connected layer:
```
Output = Activation(Input × Weights + Bias)
```

---

### Higher-Dimensional Matrices

**Why We Need More Dimensions:**
- Real data has more structure
- An RGB image isn't just width×height—it has color channels (Red, Green, Blue)
- We often process multiple images at once (batch processing)

---

While matrix operations are fundamental, real-world AI models work with higher-dimensional data structures called tensors. 
- This is because we often process batches of multi-channel data (like images or sequences).

---

### What Are Tensors? The Basic Data Structure of AI

In mathematics and computer science, a **tensor** is a generalization of vectors and matrices to higher dimensions. 
- Think of it as a container for numerical data that can have multiple dimensions.

---

**Simple Analogy:**
- **Scalar**: A single number (0-dimensional tensor)  
  Example: `temperature = 25`
- **Vector**: A list of numbers (1-dimensional tensor)  
  Example: `position = [x, y, z]`
- **Matrix**: A 2D grid of numbers (2-dimensional tensor)  
  Example: `grayscale_image = [[0.1, 0.2], [0.3, 0.4]]`
- **3D Tensor**: A cube of numbers  
  Example: `RGB_image[height][width][color_channels]`
- **Higher-Dimensional Tensors**: Used for batches of data, sequences, etc.

---

**Why Tensors Matter in AI:**
- Neural networks process data as tensors
- Each layer transforms input tensors to output tensors
- Operations like convolution, matrix multiplication, and activation functions work on tensors

---

**Tensor Examples:**
```
# A single grayscale image (28×28 pixels)
Matrix: [28][28]

# A batch of 32 RGB images (32×224×224×3)
Tensor: [32, 224, 224, 3]
  Dimension 0: Batch size (32 images)
  Dimension 1: Height (224 pixels)
  Dimension 2: Width (224 pixels)  
  Dimension 3: Color channels (3: Red, Green, Blue)
```

---

**Tensor Operations Example:**
```python
# Simple tensor operations (conceptual)
input_tensor = [               # Batch of 2 images, each 2x2 pixels, 3 colors
    [[[1, 2, 3], [4, 5, 6]],   # Image 1
     [[7, 8, 9], [10, 11, 12]]],
    
    [[[13, 14, 15], [16, 17, 18]],   # Image 2
     [[19, 20, 21], [22, 23, 24]]]
]
# Shape: [2, 2, 2, 3] = [batch_size, height, width, channels]
```

---

#### Some Common Tensor Operations in AI:
1. **Element-wise operations**: Apply function to each element
2. **Matrix multiplication**: Dot product of matching dimensions
3. **Convolution**: Slide filters across spatial dimensions
4. **Reduction**: Sum/max/mean across dimensions
5. **Reshaping**: Change tensor dimensions without changing data

---

 **1. Element-wise Operations**
**Apply a function to every element individually**

```python
import torch

# Create two tensors
A = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

B = torch.tensor([[2.0, 2.0, 2.0],
                  [2.0, 2.0, 2.0]])

# Element-wise addition
C = A + B
print("Element-wise addition:")
print(C)
# Output: [[3., 4., 5.], [6., 7., 8.]]

# Element-wise multiplication
D = A * B
print("\nElement-wise multiplication:")
print(D)
# Output: [[2., 4., 6.], [8., 10., 12.]]

# Element-wise function application
E = torch.exp(A)  # Apply exponential function to each element
print("\nElement-wise exponential:")
print(E)
# Output: [[2.7183, 7.3891, 20.0855], [54.5982, 148.4132, 403.4288]]
```

**Real AI Example:** Activation functions like ReLU
```python
# ReLU: Sets all negative values to 0
x = torch.tensor([[-1.0, 2.0, -3.0],
                  [4.0, -5.0, 6.0]])

relu_output = torch.relu(x)  # Element-wise max(0, x)
print("ReLU activation:")
print(relu_output)
# Output: [[0., 2., 0.], [4., 0., 6.]]
```

---

**2. Matrix Multiplication: Dot product of matching dimensions**

```python
# Create two matrices
X = torch.tensor([[1.0, 2.0],   # Shape: 2x2
                  [3.0, 4.0]])

W = torch.tensor([[2.0, 0.0],   # Shape: 2x2
                  [1.0, 2.0]])

# Matrix multiplication
Y = torch.matmul(X, W)  # or X @ W
print("Matrix multiplication:")
print(Y)
```

---


```
Output: [[4., 4.], [10., 8.]]
Calculation:
  Y[0,0] = 1*2 + 2*1 = 4
  Y[0,1] = 1*0 + 2*2 = 4
  Y[1,0] = 3*2 + 4*1 = 10
  Y[1,1] = 3*0 + 4*2 = 8
```

---

**Different dimensions example**
```python
A = torch.tensor([[1.0, 2.0, 3.0],  # Shape: 2x3
                  [4.0, 5.0, 6.0]])

B = torch.tensor([[1.0, 2.0],        # Shape: 3x2
                  [3.0, 4.0],
                  [5.0, 6.0]])

C = A @ B  # Shape: 2x2
print("\nMatrix multiplication with different shapes:")
print(C)
# Output: [[22., 28.], [49., 64.]]
```

---

**Real AI Example:** Fully connected layer (dense layer)
```python
# Input features: 3 samples, each with 4 features
input_data = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                           [2.0, 3.0, 4.0, 5.0],
                           [3.0, 4.0, 5.0, 6.0]])  # Shape: 3x4
# Weight matrix: transform 4 features to 3 outputs
weights = torch.tensor([[0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                        [1.0, 1.1, 1.2]])  # Shape: 4x3
# Bias vector
bias = torch.tensor([0.1, 0.2, 0.3])
# Fully connected layer: y = xW + b
output = torch.matmul(input_data, weights) + bias
print("\nFully connected layer output:")
print(output)  # Shape: 3x3
```

---

**Convolution**
**Slide filters across spatial dimensions**

```python
# Create an image (1 image, 1 color channel, 5x5 pixels)
image = torch.tensor([
    [[[1.0, 2.0, 3.0, 4.0, 5.0],
      [6.0, 7.0, 8.0, 9.0, 10.0],
      [11.0, 12.0, 13.0, 14.0, 15.0],
      [16.0, 17.0, 18.0, 19.0, 20.0],
      [21.0, 22.0, 23.0, 24.0, 25.0]]]
])  # Shape: [1, 1, 5, 5] = [batch, channels, height, width]

# Create a filter (1 input channel, 1 output channel, 3x3 kernel)
filter = torch.tensor([
    [[[1.0, 0.0, -1.0],
      [1.0, 0.0, -1.0],
      [1.0, 0.0, -1.0]]]
])  # Shape: [1, 1, 3, 3] = [output_channels, input_channels, height, width]

# Apply convolution
output = torch.nn.functional.conv2d(image, filter, padding=1)
print("Convolution output:")
print(output)
# The filter slides over the image, computing dot products at each position
```

---

**Simpler 1D convolution example:**
```python
# 1D signal (like audio or time series data)
signal = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
signal = signal.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 8]

# 1D filter (detecting edges)
kernel = torch.tensor([[1.0, -1.0]]).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 2]

# Convolution
result = torch.nn.functional.conv1d(signal, kernel)
print("\n1D convolution (edge detection):")
print(result)
# Output: [[[-1., -1., -1., -1., -1., -1., -1.]]]
# Each output is difference between consecutive inputs
```

---

**4. Reduction Operations**
**Sum/max/mean across dimensions**

```python
# Create a tensor
tensor = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])

# Sum all elements
total_sum = torch.sum(tensor)
print(f"Sum of all elements: {total_sum}")  # 45.0

# Sum across rows (dimension 0)
column_sums = torch.sum(tensor, dim=0)
print(f"\nSum of each column (dim=0): {column_sums}")
# Output: [12., 15., 18.] = [1+4+7, 2+5+8, 3+6+9]

# Sum across columns (dimension 1)
row_sums = torch.sum(tensor, dim=1)
print(f"Sum of each row (dim=1): {row_sums}")
# Output: [6., 15., 24.] = [1+2+3, 4+5+6, 7+8+9]

# Mean across rows
column_means = torch.mean(tensor, dim=0)
print(f"\nMean of each column: {column_means}")
# Output: [4., 5., 6.] = [12/3, 15/3, 18/3]

# Maximum value in each column
column_max = torch.max(tensor, dim=0)
print(f"Max of each column: {column_max.values}")
# Output: [7., 8., 9.]

# Global maximum
global_max = torch.max(tensor)
print(f"Global maximum: {global_max}")  # 9.0

# Softmax (common in classification)
scores = torch.tensor([2.0, 1.0, 0.1])
probabilities = torch.softmax(scores, dim=0)
print(f"\nSoftmax probabilities: {probabilities}")
# Output: [0.6590, 0.2424, 0.0986] (sums to 1)
```

---

**5. Reshaping Operations**
**Change tensor dimensions without changing data**

```python
# Create a 1D tensor
original = torch.arange(12)  # [0, 1, 2, ..., 11]
print(f"Original tensor: {original}")
print(f"Original shape: {original.shape}")  # torch.Size([12])

# Reshape to 2D
reshaped_2d = original.reshape(3, 4)
print("\nReshaped to 3x4:")
print(reshaped_2d)
# Output: [[0, 1, 2, 3],
#          [4, 5, 6, 7],
#          [8, 9, 10, 11]]

# Reshape to 3D
reshaped_3d = original.reshape(2, 3, 2)
print("\nReshaped to 2x3x2:")
print(reshaped_3d)
print(f"Shape: {reshaped_3d.shape}")  # torch.Size([2, 3, 2])

# Transpose (swap dimensions)
matrix = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]])
transposed = matrix.T  # or torch.transpose(matrix, 0, 1)
print("\nOriginal matrix:")
print(matrix)  # 2x3
print("\nTransposed matrix:")
print(transposed)  # 3x2

# Flatten (convert to 1D)
flattened = matrix.flatten()
print(f"\nFlattened: {flattened}")
# Output: [1., 2., 3., 4., 5., 6.]

# View (similar to reshape but shares memory)
viewed = original.view(4, 3)
print("\nUsing view (4x3):")
print(viewed)

# Squeeze/Unsqueeze (add/remove dimensions of size 1)
tensor_3d = torch.randn(1, 3, 1, 5)  # Shape: [1, 3, 1, 5]
squeezed = torch.squeeze(tensor_3d)  # Remove all size-1 dims
print(f"\nOriginal shape: {tensor_3d.shape}")
print(f"Squeezed shape: {squeezed.shape}")  # [3, 5]

# Add batch dimension (common in neural networks)
single_image = torch.randn(3, 224, 224)  # [channels, height, width]
batched = single_image.unsqueeze(0)  # Add batch dimension
print(f"\nSingle image shape: {single_image.shape}")
print(f"Batched shape: {batched.shape}")  # [1, 3, 224, 224]
```

---

**Complete Example: Simple Neural Network Layer**

```python
# Putting it all together: A complete neural network layer
def neural_network_layer(x, W, b):
    """
    x: input tensor of shape [batch_size, input_features]
    W: weight matrix of shape [input_features, output_features]
    b: bias vector of shape [output_features]
    """
    # Matrix multiplication: xW
    z = torch.matmul(x, W)
    # Add bias (broadcasting happens automatically)
    z_plus_bias = z + b
    # Apply ReLU activation (element-wise)
    a = torch.relu(z_plus_bias)
    return a

# Example usage
batch_size = 4
input_features = 10
output_features = 5

# Create random inputs and parameters
x = torch.randn(batch_size, input_features)
W = torch.randn(input_features, output_features)
b = torch.randn(output_features)

# Forward pass
output = neural_network_layer(x, W, b)
print(f"Input shape: {x.shape}")
print(f"Weight shape: {W.shape}")
print(f"Output shape: {output.shape}")
# Output shape: [4, 5] (batch_size, output_features)
```

---

### Common Pitfalls to Avoid

```python
# 1. Dimension mismatch in matrix multiplication
A = torch.randn(3, 4)  # 3x4
B = torch.randn(2, 4)  # 2x4
# C = A @ B  # ERROR! Inner dimensions don't match (4 vs 2)

# 2. Broadcasting might not do what you expect
A = torch.randn(3, 4, 5)
B = torch.randn(5, 6)
# C = A + B  # ERROR! Shapes not broadcastable

# 3. Reshape only works if total elements match
tensor = torch.randn(12)
# reshaped = tensor.reshape(3, 5)  # ERROR! 12 elements can't make 3x5 (needs 15)

# 4. Views require contiguous memory
non_contiguous = tensor[::2]  # Take every other element
# viewed = non_contiguous.view(3, 2)  # ERROR! Tensor not contiguous
reshaped = non_contiguous.reshape(3, 2)  # OK! Reshape works
```

---


- These examples show how fundamental tensor operations form the building blocks of all AI models. 
- Each neural network layer is just a combination of these basic operations!

---

## The Hardware Evolution: From Scalar to Tensor Processing

---

### 1. Scalar Processors: The General-Purpose Workhorse

**How CPUs Work (Sequential Processing)**
- Imagine you're a single worker in a factory. 
  - You receive one item, 
  - process it completely, 
  - then move to the next item.


---


**CPU Architecture:**
```
CPU Core 1: [ALU] [Control Unit] [Cache] [Registers]
CPU Core 2: [ALU] [Control Unit] [Cache] [Registers]
CPU Core 3: [ALU] [Control Unit] [Cache] [Registers]
CPU Core 4: [ALU] [Control Unit] [Cache] [Registers]
```

**Key Features:**
- **ALU (Arithmetic Logic Unit)**: Processes one or two numbers at a time
- **Complex Control**: Can handle branches, loops, and different instructions
- **Sophisticated Cache**: Multiple cache levels (L1, L2, L3) to hide memory latency
- **Clock Speed**: Modern CPUs run at 3-5 GHz (3-5 billion cycles per second)


---

**Example: Matrix Addition on CPU**
```python
# Python/NumPy example - how a CPU might process this
import numpy as np

# Create two 1024×1024 matrices
A = np.random.randn(1024, 1024)  # 1 million elements
B = np.random.randn(1024, 1024)

# Traditional CPU approach (conceptually):
C = np.zeros((1024, 1024))
for i in range(1024):          # Outer loop
    for j in range(1024):      # Inner loop
        # Process ONE element at a time
        C[i][j] = A[i][j] + B[i][j]
        
# Total operations: 1,048,576 additions
# At 3 GHz: Could theoretically do 3 billion operations per second
# But reality: Memory access, cache misses, branch prediction slow it down
```

---

**CPU Limitation for AI: The Memory Wall**
**Matrix Multiplication: C = A × B**
- 1024×1024 matrices have 1,048,576 elements each
- Naive algorithm requires ~1 billion multiply-add operations
- 3 GHz CPU: Could theoretically do this in ~0.33 seconds
- Reality: ~5-10 seconds due to:
  * Memory bandwidth limits (can't feed data fast enough)
  * Cache misses (data not in fast cache memory)
  * Instruction overhead (loop counters, branches)

---


**Exercise: Actual Performance:**
```python
import time
import numpy as np

# Test with NumPy (which uses optimized C code)
A = np.random.randn(1024, 1024).astype(np.float32)
B = np.random.randn(1024, 1024).astype(np.float32)

start = time.time()
C = np.matmul(A, B)  # NumPy uses optimized BLAS libraries
cpu_time = time.time() - start

print(f"1024×1024 matrix multiplication on CPU: {cpu_time:.3f} seconds")
# Typical result: 0.1-0.5 seconds (with optimized libraries)
# Without optimization: Could be 10+ seconds!
```

---



### 2. Vector Processors: Doing Many Things at Once

**SIMD (Single Instruction, Multiple Data)**
Imagine you're now a worker with 8 hands. You can process 8 items simultaneously!


---

**CPU SIMD Extensions:**
- **SSE** (Streaming SIMD Extensions): Process 4 floats at once
- **AVX** (Advanced Vector Extensions): Process 8 floats at once
- **AVX-512**: Process 16 floats at once

```python
# SIMD concept (pseudocode)
# Instead of:
for i in range(8):
    c[i] = a[i] + b[i]

# SIMD does:
c[0:7] = a[0:7] + b[0:7]  # All 8 additions in ONE instruction
```

---

CPUs are general-purpose but limited by sequential processing and memory bottlenecks. For AI workloads, we need parallelism.

---

**GPU Architecture: Massive Parallelism**
Imagine you have 10,000 workers, each doing the same simple task on different data.

**GPU vs CPU Architecture:**
```
CPU: 4-64 complex cores
      [Core1] [Core2] [Core3] [Core4]
        ↓       ↓       ↓       ↓
    [Shared L3 Cache] → [Main Memory]

GPU: 1000s of simple cores organized in groups
      [SM1: 64 cores] [SM2: 64 cores] ... [SMN: 64 cores]
           ↓                ↓                     ↓
      [Shared Memory]  [Shared Memory]    [Shared Memory]
           ↓                ↓                     ↓
                  [Global GPU Memory]
```

---

**GPU Memory Hierarchy:**
```
Registers (fastest, per-thread)      <-- Each thread has its own
     ↓
Shared Memory (fast, per-block)      <-- Threads in block share
     ↓
Global Memory (slow, all threads)    <-- All threads can access
```

---

**CUDA Example: Vector Addition on GPU**
```python
# PyTorch example - shows GPU advantage
import torch
import time

# Create large tensors
size = 10000000  # 10 million elements
a_cpu = torch.randn(size)
b_cpu = torch.randn(size)

# CPU version
start = time.time()
c_cpu = a_cpu + b_cpu
cpu_time = time.time() - start

# GPU version
a_gpu = a_cpu.cuda()  # Move to GPU
b_gpu = b_cpu.cuda()

# GPU warm-up
torch.cuda.synchronize()
start = time.time()
c_gpu = a_gpu + b_gpu
torch.cuda.synchronize()  # Wait for GPU to finish
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
# Typical result: GPU is 10-50x faster for large operations
```

---

**Why GPUs Excel at Matrix Multiplication:**

Matrix multiplication pattern
```python
For each output element C[i][j]:
  C[i][j] = sum(A[i][k] * B[k][j] for k in range(N))
```
GPU can compute MANY output elements in parallel:
```
Thread 1: Compute C[0][0]
Thread 2: Compute C[0][1]
Thread 3: Compute C[0][2]
...
Thread 1,048,576: Compute C[1023][1023]
```
ALL compute simultaneously!


---

GPUs provide massive parallelism but are still general-purpose. The next step is specialization for tensor operations.

---

### 3. Tensor Processors: Specialized for AI

**The Matrix Multiplication Unit (Systolic Array)**
- Imagine a factory assembly line specifically designed for matrix multiplication. 
- Data flows through in a rhythmic pattern.


---

**Google TPU Architecture:**
```
Input:     A matrix      B matrix
              ↓             ↓
        [Systolic Array: 256×256 MAC units]
              ↓
        [Accumulators]
              ↓
        [Activation Unit]
              ↓
        Output: C matrix
```

---

**How a 4×4 Systolic Array Works:**
```
Time 1: Load first elements
[ a00*b00 ] [  0   ] [  0   ] [  0   ]
[   a01   ] [ a00  ] [  0   ] [  0   ]
[   a02   ] [  0   ] [ a00  ] [  0   ]
[   a03   ] [  0   ] [  0   ] [ a00  ]

Time 2: Shift and multiply
[ a00*b00 + a01*b10 ] [ a01*b11 ] [  0   ] [  0   ]
[   a02   ] [ a00*b01 + a01*b11 ] [ a01  ] [  0   ]
[   a03   ] [  0   ] [ a00*b02 + a01*b12 ] [ a01  ]
[   a04   ] [  0   ] [  0   ] [ a00*b03 + a01*b13 ]

... continues until full matrix computed
```

---

**NVIDIA Tensor Cores: Mixed-Precision Power**

- Tensor Cores compute: D = A × B + C
  - Where A, B, C, D are small matrices (like 4×4 or 16×16)

- Regular GPU core: 1 multiply-add per cycle (FP32)
- Tensor Core: 64 multiply-adds per cycle (mixed precision)

Example: A100 GPU has 432 Tensor Cores
 - Each can do 64 FP16 operations per cycle
 - Total: 432 × 64 = 27,648 operations per cycle!


---

**Performance Comparison Example**

**Matrix Multiplication Performance (1024×1024)**

| Hardware | Performance (TFLOPS) | Time for 1024×1024 MatMul | Power Consumption |
|----------|----------------------|---------------------------|-------------------|
| **CPU (Intel i9)** | 0.5-1 TFLOPS (FP32) | ~1.0 second | 65-125W |
| **GPU (NVIDIA RTX 4090)** | 80 TFLOPS (FP32) | ~0.02 seconds | 450W |
| **GPU with Tensor Cores (A100)** | 312 TFLOPS (FP16 Tensor) | ~0.0005 seconds | 400W |
| **TPU v4** | 275 TFLOPS (BF16) | ~0.0002 seconds | ~250W |


---

**Key Observations:**

1. **Performance Gap**: Tensor processors (A100, TPU) are 500-5000x faster than CPUs for matrix multiplication
2. **Power Efficiency**: TPUs achieve higher performance with less power than high-end GPUs
3. **Specialization**: The more specialized the hardware, the better the performance for AI workloads
4. **Precision Trade-offs**: Tensor Cores/TPUs use lower precision (FP16/BF16) for speed, which is acceptable for most AI applications


---

### Real Hardware Examples

**Exercise 1. Apple Neural Engine (iPhone)**
```python
# On-device AI example
import torch
import torchvision
import time

# Load a model
model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()

# Create sample input
input_tensor = torch.randn(1, 3, 224, 224)

# Run on CPU
start = time.time()
output_cpu = model(input_tensor)
cpu_time = time.time() - start

# Run on Neural Engine (if available)
if torch.backends.mps.is_available():
    model = model.to('mps')
    input_tensor = input_tensor.to('mps')
    start = time.time()
    output_gpu = model(input_tensor)
    torch.mps.synchronize()
    gpu_time = time.time() - start
    print(f"Neural Engine speedup: {cpu_time/gpu_time:.1f}x")
```

---


**Exercise 2. Mixed Precision in Practice**
```python
# Using Tensor Cores with mixed precision
import torch

# Automatic mixed precision (AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# Inside training loop:
with autocast():  # Uses Tensor Cores for FP16 operations
    outputs = model(inputs)  # Runs in mixed precision
    loss = criterion(outputs, targets)

# Backward pass
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

### Hardware Specialization Spectrum

```
General Purpose  →  Specialized
      ↓                ↓
   CPU           →   GPU    →   TPU    →   ASIC
   (All tasks)      (Parallel)  (Matrix)   (One task)
   
Examples:
CPU: Intel Core, AMD Ryzen
GPU: NVIDIA GeForce, AMD Radeon
TPU: Google TPU, NVIDIA Tensor Cores
ASIC: Groq Chip, Cerebras WSE
```

---

### Why This Matters for AI Compilers

**The Compiler's Challenge:**
```python
# Same PyTorch code needs to run on different hardware:
model = MyNeuralNetwork()
# The compiler must generate different code for:
if target == "CPU":
    # Use AVX-512 instructions, optimize for cache
    generate_cpu_code(model)
elif target == "GPU":
    # Use CUDA threads, shared memory, Tensor Cores
    generate_gpu_code(model)  
elif target == "TPU":
    # Map to systolic array, optimize data flow
    generate_tpu_code(model)
elif target == "iPhone":
    # Use Neural Engine, optimize for power
    generate_ane_code(model)
```

---

**Key Takeaway:**
- **CPUs**: Flexible but slower for tensor operations
- **GPUs**: Excellent for parallel operations, good balance
- **TPUs/Tensor Cores**: Specialized for matrix math, fastest for AI
- **AI Compilers**: Must understand ALL these architectures to generate optimal code

The hardware dictates what optimizations are possible, and the compiler's job is to map the computational graph to the hardware's strengths.

---


## The AI Compiler: Bridging Models and Hardware


### The AI Compilation Pipeline

**Step 1: Model to Computational Graph**
- AI frameworks (PyTorch, TensorFlow) define models as operations
- Compiler extracts a directed graph of computations
- Nodes: Operations (matmul, conv, relu)
- Edges: Data dependencies (tensors)


---

**Example Computational Graph:**
```
      [Input Image]
          ↓
    [Convolution]
          ↓
   [Batch Normalization]
          ↓
      [ReLU]
          ↓
    [Max Pooling]
          ↓
     [Output]
```

---

**Step 2: Graph-Level Optimizations**
- **Operator fusion**: Combine multiple operations into one
  Example: Conv → BatchNorm → ReLU → FusedConvBNReLU
- **Constant folding**: Pre-compute constant expressions
- **Dead code elimination**: Remove unused operations
- **Layout transformations**: NHWC ↔ NCHW data format transformation  for hardware efficiency


---


**Step 3: Hardware-Specific Lowering**
- Transform graph to hardware-specific operations
- Choose optimal kernels from database
- Generate efficient memory access patterns


---


**Step 4: Code Generation**
- Generate actual executable code
- CUDA for NVIDIA GPUs
- Metal for Apple devices
- Vulkan for cross-platform GPU
- LLVM for CPUs


---

### AI Compiler Ecosystem in 2025

#### Major Frameworks and Trends

**1. TVM (Apache TVM) - The End-to-End AI Compiler**
- **Purpose**: Compile models from multiple frameworks (PyTorch, TensorFlow, ONNX) to diverse hardware
- **Key Innovation**: AutoTVM - uses machine learning to automatically find optimal operator schedules
- **Target Support**: CPU, GPU (CUDA, ROCm, Metal), TPU, FPGA, and custom accelerators
- **Educational Value**: Implements the full compilation pipeline you're learning about

---

```
Frontend Models
(PyTorch, TF, ONNX, etc.)
    ↓
Relay IR (High-level)
    ↓
Graph Optimizations
(Fusion, Constant folding, etc.)
    ↓
TIR (Tensor IR) - Loop-level
    ↓
AutoTVM / AutoScheduler
(Finds optimal schedules)
    ↓
Hardware-Specific Codegen
(LLVM, CUDA, Metal, Vulkan, etc.)
    ↓
Deployable Module
(Runs on CPU/GPU/TPU/FPGA/etc.)
```

---

**Key Components:**
1. **Relay**: High-level IR for graph optimizations
2. **TIR**: Low-level IR for loop transformations
3. **AutoTVM**: Uses ML to find optimal operator implementations
4. **Runtime**: Lightweight deployment across platforms
   
---



**2. XLA (Accelerated Linear Algebra) - Google's Compiler**
- **Primary Use**: JIT compilation for TensorFlow and JAX
- **Specialization**: Excellent for TPU optimization
- **Integration**: Built into TensorFlow, used automatically in many cases

---

**3. MLIR (Multi-Level Intermediate Representation)**
- **Purpose**: Not a compiler itself, but infrastructure for building compilers
- **Innovation**: "Dialect" system allows different abstraction levels (tensor ops, low-level loops, hardware ops)
- **Used By**: Many newer compilers build on MLIR (like IREE for mobile, Circt for hardware)

---

**4. TensorRT - NVIDIA's Inference Optimizer**
- **Focus**: Maximum performance on NVIDIA GPUs
- **Features**: Layer fusion, precision calibration, kernel auto-tuning
- **Limitation**: NVIDIA-only, inference-only

---

**2025 Trends and Developments:**

**1. Unified Multi-Backend Support**
- Single model compilation to CPU, GPU, TPU, NPU, FPGA
- Automatic backend selection based on available hardware
- Dynamic switching between backends at runtime


---

**2. Advanced Auto-Tuning and AI-for-Compilers**
- Machine learning to predict optimal optimizations
- Reinforcement learning for schedule discovery
- Neural networks that learn to optimize neural networks


---

**3. Dynamic Shape and Sparsity Support**
- Better handling of variable-sized inputs
- Automatic exploitation of sparse tensors
- Compile-time/runtime hybrid approaches


---

**4. Privacy-Preserving Compilation**
- Federated learning optimizations
- Secure multi-party computation compilation
- Differential privacy guarantees at compile time


---


**5. Energy-Aware Compilation**
- Optimizing for battery life on mobile devices
- Thermal-aware scheduling
- Carbon footprint reduction through compiler optimizations


---


**6. Specialized Hardware Proliferation**
- Domain-specific accelerators for vision, NLP, recommendation
- In-memory computing compilation
- Optical computing interfaces

---

### Real-World Impact: Case Study

**Before AI Compilers:**
- Researchers hand-tuned CUDA kernels for each new model
- Months of optimization for production deployment
- Hardware-specific code that couldn't run elsewhere
- Inference latency: 100ms, Power consumption: 10W

---

**With Modern AI Compilers:**
- Write model once in PyTorch/TensorFlow
- Compiler automatically optimizes for target hardware
- Hours instead of months for deployment
- Inference latency: 10ms, Power consumption: 2W
- Same model runs on cloud GPU, mobile phone, or edge device

---

### Hands-On: From Tensor Operations to Compiled Code

**Simple Example: Matrix Multiplication**
**1. Define in PyTorch (high-level)**

```python
import torch
A = torch.randn(1024, 1024)
B = torch.randn(1024, 1024)
C = torch.matmul(A, B)  # This is a tensor operation
```

---

**2. What the compiler sees (computational graph)**
 
> MatMul(A, B) -> C

---

**3. Possible optimizations**
   - Tiling: Break 1024×1024 into 32×32 blocks
   - Vectorization: Use SIMD instructions within blocks
   - Parallelization: Process blocks on multiple GPU cores
   - Memory layout: Convert to column-major for better cache usage

---

**4. Generated code (conceptual CUDA):**
```python
"""
__global__ void matmul_kernel(float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];
    
    float sum = 0;
    for (int tile = 0; tile < 1024/32; tile++) {
        // Load tiles from global to shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row*1024 + tile*32 + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(tile*32+threadIdx.y)*1024 + col];
        __syncthreads();
        
        // Compute partial product
        for (int k = 0; k < 32; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row*1024 + col] = sum;
}
"""
```

---

### AI Framework Comparison: PyTorch vs JAX vs TensorFlow

#### PyTorch Example: Dynamic and User-Friendly
```python
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Instantiate and run
model = SimpleNet()
x = torch.randn(32, 784)  # Batch of 32 images
output = model(x)

# JIT compilation for deployment
traced_model = torch.jit.trace(model, x)
traced_model.save("model.pt")  # Deploy anywhere
```

---

**PyTorch Compilation Flow:**
```
PyTorch Model → TorchScript → Optimizations → Hardware-specific code
```

---

### JAX Example: Functional and Composable
```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

# Pure functions for neural network
def predict(params, inputs):
    # Unpack parameters
    W1, b1, W2, b2 = params
    
    # Forward pass
    hidden = jnp.dot(inputs, W1) + b1
    hidden = jnp.maximum(0, hidden)  # ReLU
    outputs = jnp.dot(hidden, W2) + b2
    return outputs

# Initialize parameters
key = jax.random.PRNGKey(0)
W1 = jax.random.normal(key, (784, 128))
b1 = jax.random.normal(key, (128,))
W2 = jax.random.normal(key, (128, 10))
b2 = jax.random.normal(key, (10,))
params = (W1, b1, W2, b2)

# JIT compile the entire function
compiled_predict = jit(predict)

# Use it (automatically uses XLA compiler)
x = jax.random.normal(key, (32, 784))
output = compiled_predict(params, x)

# Also get gradients automatically
grad_fn = jit(grad(lambda p, x: predict(p, x).mean()))
gradients = grad_fn(params, x)
```

---

**JAX Compilation Flow:**
```
JAX Functions → XLA HLO IR → Optimizations → CPU/GPU/TPU code
```

---

#### TensorFlow Example: Production-Ready
```python
import tensorflow as tf

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])

# Convert to TensorFlow Lite for mobile
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save for deployment
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

**TensorFlow Compilation Flow:**
```
Keras Model → TensorFlow Graph → XLA → TFLite/TPU/GPU code
```

---

### From Framework Code to Optimized Hardware

#### Multi-Framework Compilation Example

**Same Model, Different Frameworks:**
```python
# PyTorch version
import torch
def torch_model(x, W1, b1, W2, b2):
    h = torch.relu(x @ W1 + b1)
    return h @ W2 + b2

# JAX version  
import jax.numpy as jnp
def jax_model(x, W1, b1, W2, b2):
    h = jnp.maximum(0, x @ W1 + b1)
    return h @ W2 + b2

# TensorFlow version
import tensorflow as tf
def tf_model(x, W1, b1, W2, b2):
    h = tf.nn.relu(tf.matmul(x, W1) + b1)
    return tf.matmul(h, W2) + b2
```

---

**What the AI Compiler Sees (Unified Computational Graph):**
```
      [Input x]
         ↓
     MatMul(x, W1)
         ↓
      Add(+ b1)
         ↓
      ReLU()
         ↓
     MatMul(h, W2)
         ↓
      Add(+ b2)
         ↓
     [Output]
```

---

**Compiler Optimizations Applied:**
```python
# 1. Operator Fusion: Combine operations
# Before: MatMul → Add → ReLU → MatMul → Add
# After:  FusedMatMulAddReLU → FusedMatMulAdd

# 2. Memory Optimization: Reuse buffers
# Before: Allocate memory for each intermediate tensor
# After:  Reuse memory across operations

# 3. Hardware-Specific Optimizations:
if target == "NVIDIA_GPU":
    # Use Tensor Cores for FP16 matmul
    # Configure CUDA thread blocks
    # Optimize shared memory usage
elif target == "Apple_Neural_Engine":
    # Quantize to 8-bit integers
    # Use AMX instructions
    # Optimize for power efficiency
elif target == "Google_TPU":
    # Map to systolic array
    # Optimize for batch processing
    # Use bfloat16 precision
```

---

**Generated Code for Different Targets:**

```c
// NVIDIA GPU (CUDA) - Optimized Kernel
__global__ void fused_kernel(float* x, float* W1, float* b1, 
                             float* W2, float* b2, float* output) {
    // Shared memory tiles
    __shared__ float tile_x[32][32];
    __shared__ float tile_W1[32][32];
    
    // Optimized for Tensor Cores
    float16 acc = 0;
    for (int tile = 0; tile < N_TILES; tile++) {
        // Load tiles
        load_tile_to_shared(x, tile_x, tile);
        load_tile_to_shared(W1, tile_W1, tile);
        __syncthreads();
        
        // Tensor Core matrix multiply-add
        acc = wmma::mma_sync(tile_x, tile_W1, acc);
        __syncthreads();
    }
    // Store result with ReLU activation
    output[threadIdx] = max(0.0f, acc + b1[threadIdx]);
}
```

---

```c
// Apple GPU (Metal) - Optimized Shader
kernel void fused_shader(
    texture2d_array<half, access::read> input [[texture(0)]],
    texture2d_array<half, access::read> weights [[texture(1)]],
    device half* bias [[buffer(0)]],
    texture2d_array<half, access::write> output [[texture(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Use Apple Neural Engine capabilities
    half4 acc = 0;
    for (int k = 0; k < K; k++) {
        half4 a = input.read(gid.xy, k);
        half4 b = weights.read(uint2(k, gid.z), 0);
        acc += a * b;
    }
    // Fused activation and bias
    acc = max(acc + bias[gid.z], 0.0h);
    output.write(acc, gid.xy, gid.z);
}
```

---


## Why This Matters for You
**Transferable Concepts You Already Know:**
Traditional Compiler Concepts → AI Compiler Applications
>
    "Lexical Analysis": "Parse model definition into tokens",
    "Parsing": "Build computational graph from operations",
    "Type Checking": "Verify tensor shapes and dtypes",
    "Control Flow Analysis": "Analyze branching in dynamic models",
    "Dataflow Analysis": "Optimize tensor memory usage",
    "Intermediate Representations": "Computational graphs as IR",
    "Optimization Passes": "Graph transformations (fusion, etc.)",
    "Code Generation": "Hardware-specific kernel generation",
    "Register Allocation": "Tensor memory allocation strategies"

---


**Past, Present, and Future**

**Historical Perspective:**
```
1970s: C compiler enabled software revolution
       → Portable code across different CPUs
       
2000s: JIT compilers enabled web revolution  
       → JavaScript runs everywhere
       
2020s: AI compilers enabling AI revolution
       → Neural networks run everywhere
```

---

**Current Trends (2024-2025):**
1. **Unified Compilation**: Single model → CPU/GPU/TPU/FPGA
2. **Automatic Optimization**: AI that compiles AI models
3. **Privacy-Preserving Compilation**: Secure ML on untrusted hardware
4. **Sustainable AI**: Energy-aware compilation techniques


---



## Further Learning Path

**If you want to explore more:**

1. **Start Simple**: 
   - Play with tensor operations in NumPy
   - Understand how GPUs accelerate these operations

2. **Hands-On Compilers**:
   - Try TVM tutorial: Compile a small neural network
   - Experiment with different optimization levels

3. **Dive Deeper**:
   - Read about MLIR and its dialect system
   - Explore how TensorRT optimizes models
   - Study polyhedral compilation for loop optimizations

---

### From Here to AI Compiler Expert

**Level 1: Foundation** 
1. Master tensor operations
import numpy as np
import torch
import jax.numpy as jnp

2. Understand GPU basics
   - CUDA programming model
   - Memory hierarchy (global/shared/registers)
   - Thread blocks and warps

3. Learn one framework deeply
   Choose: PyTorch, JAX, or TensorFlow

---

**Level 2: Basics**

1. Study traditional compilers (you're doing this!)
   - Lexing, parsing, IR, optimizations, codegen

2. Explore AI compiler tools
   - TVM: pip install apache-tvm
   - MLIR: https://mlir.llvm.org/
   - ONNX Runtime: pip install onnxruntime

3. Simple project: Optimize a small model

---

**Level 3: Advanced Topics** 

 1. Kernel optimization
    - Polyhedral compilation
   - Auto-tuning
    - Memory access patterns
  
---

 1. Hardware-specific optimization
    - GPU (CUDA, ROCm, Metal)
   - TPU (XLA optimizations)
    - Edge devices (TensorFlow Lite)


---


3. Build a simple AI compiler
    - Graph IR implementation
    - Optimization passes
    - Code generation for a simple target
  
---


**Level 4: Specialization**
> "Performance": "Extreme optimization for specific hardware",
    "Correctness": "Formal verification of AI compilers",
    "Productivity": "Better developer tools and debuggers",
    "Accessibility": "Compilers for novel hardware (quantum, optical)",
    "Sustainability": "Energy-efficient compilation techniques"


---

**Key takeaway**: The concepts you learned in this course—parsing, IRs, optimizations, code generation—are exactly what power AI compilers. 