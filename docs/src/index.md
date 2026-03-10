
## JACC Introduction

JACC.jl is a Julia package for performance portable CPU/GPU kernels using metaprogramming on top of existing Julia backends. It enables users to write a single code based on an **`array`** memory model and **`parallel_for`** and **`parallel_reduce`** functions that can run on both CPUs and GPUs without needing to rewrite the code for each platform.

!!! why "Who can benefit with JACC?"

    Scientists and engineers can benefit from JACC's default basic APIs that enable minimal effort to access CPU and GPU parallel capabilities to accelerate codes and let users focus on their algorithms and science. Advanced users can also access low-level backend specific features (e.g., blocks/threads and synchronization) when needed. 


![parallel_for](assets/parallel_for.png)

As shown in the schematics, **`parallel_for`** (and **`parallel_reduce`**) constructs are functional forms of for-loops that can be executed in parallel on multiple processing units (e.g., CPU cores, GPU threads). They allow users to express parallelism in their code without dealing with low-level threading or synchronization details across vendor architectures. JACC provides a unified API to launch parallel workloads on different back ends (CPU/GPU) without changing the user's code.

While the [JuliaGPU](https://juliagpu.org/) ecosystem provides powerful tools for GPU programming, JACC sits on top of these backends leveraging these capabilities to simplify the process of writing portable code that can run on both CPUs and GPUs without modification. Thus filling a gap in the Julia ecosystem for productive, high-level programming for performance portability across heterogeneous computing platforms.

*JACC architecture overview* - application image credit: CERFACS -

![JACC Architecture](assets/jacc.png)


Resources:

- For a broader understanding of JACC's design principles and goals, please refer to the [JACC paper at SC-W 2024](https://ieeexplore.ieee.org/document/10820713) - open version [available here](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400b955/555400b955.pdf).
- For a full example using JACC, see the Gray-Scott [code](https://github.com/JuliaORNL/GrayScott.jl/blob/main/src/simulation/Simulation.jl#L11) and "Julia for HPC" Tutorial [material](https://juliaornl.github.io/TutorialJuliaHPC/)


## Why Julia?

[Julia](https://julialang.org/) enables scientists and engineers to write code more quickly and to focus on their science or technical domain. 

- Julia combines high-level/low-level capabilities for easy to read and write code due to its expressive mathematical syntax (e.g. arrays, GPU programming, parallelization). 
- Julia's just-in-time (JIT) compilation based on LLVM allows for speeds comparable to C, C++ or Fortran.
- Julia's unified package manager, [Pkg.jl](https://github.com/JuliaLang/Pkg.jl), simplifies the process of managing dependencies.
- Julia's rich ecosystem includes a wide range of libraries and tools for scientific computing, data analysis, and machine learning.


## Supported backends

- **CPU (default)**:
  - `Threads`: multi-threading on CPU using Julia's built-in threading capabilities.
- **GPU from [JuliaGPU](https://juliagpu.org/)**:
  - `CUDA`: NVIDIA GPUs using the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) package.
  - `AMDGPU`: AMD GPUs using the [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) package.
  - `Metal`: Apple Silicon GPUs using the [Metal.jl](https://github.com/JuliaGPU/Metal.jl) package.
  - `oneAPI`: Intel GPUs using the [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) package (Experimental).

## Installation

JACC is a registered Julia package. Install JACC using Julia's [Pkg.jl](https://pkgdocs.julialang.org/dev/managing-packages/#Managing-Packages) capabilities for managing packages. 
e.g.

Installation from the Julia REPL:

```
julia> ]
pkg> add JACC
```

from command line (e.g. CI):

```bash
julia -e 'using Pkg; Pkg.add("JACC")'
```


## Quick start - filling and reducing an array

Prerequisite: 
- [Julia 1.11 or later](https://julialang.org/downloads/)

To run an example from scratch

- Install JACC.jl

```bash
julia -e 'using Pkg; Pkg.add("JACC")'
```

- Copy this file to your local machine and save it as `jacc-saxpy.jl`:

```julia
import JACC
JACC.@init_backend

function axpy(i, alpha, x, y)
    @inbounds x[i] += alpha * y[i]
end

N = 10
alpha = Float32(2.0)
x = JACC.zeros(Float32, N)
y = JACC.array(fill(Float32(5), N))
JACC.@parallel_for range=N axpy(alpha, x, y)
@show x
x_sum = JACC.parallel_reduce(x)
@show x_sum
```

- Run the example using the default Threads back end with 4 threads:

```bash
julia -t 4 jacc-saxpy.jl
```

- Switch to another backend, e.g. assuming access to NVIDIA GPU and that CUDA is installed and configured:

```bash
julia -e 'using JACC; JACC.set_backend("CUDA")'
```

!!! note

    This step might take a while the first time downloading all CUDA.jl dependencies. After installation please refer to CUDA.jl docs if wanting to [use a local CUDA installation](https://cuda.juliagpu.org/stable/installation/overview/#Using-a-local-CUDA).

!!! note

    Mac users can use the "Metal" backend for `Float32` computations as `Float64 is not supported on Apple Silicon GPUs`. To set the Metal backend use `JACC.set_backend("Metal")`.


- Run the example again, but this time the same code will run on the GPU:

```bash
julia jacc-saxpy.jl
```
