## API components

JACC APIs consist of three main components:

- **Backend selection**: e.g. `JACC.set_backend("CUDA")` (outside user code) and `JACC.@init_backend` (inside user code)

- **Memory allocation**: e.g. `JACC.array{T}()`, `JACC.zeros`, `JACC.ones`, `JACC.shared`, `JACC.Multi`

- **Kernel launching**: e.g. `JACC.@parallel_for` and `JACC.@parallel_reduce` (along with functions for `JACC.parallel_for` and `JACC.parallel_reduce`)

## Backend selection

- **`JACC.set_backend`**: allows selecting the runtime backend on **CPU**: `Threads` (default) and **GPU**: `CUDA`, `AMDGPU`, `oneAPI`. Uses Preferences.jl and stores the selected backend in a [LocalPreferences.jl](https://github.com/JuliaPackaging/Preferences.jl) file if JACC.jl is a project dependency. Use `JACC.set_backend` prior to running any code targeting a particular backend.

Example:
```julia
using JACC
JACC.set_backend("CUDA")
```

or from the command line (e.g. in CI workflows):

```
$ julia -e 'using JACC; JACC.set_backend("CUDA")'
```

!!! danger 

    `set_backend` will polute your project's `Project.toml` adding the selected backend package. Beware of committing this change (e.g. during development). To clean up, you can run `JACC.unset_backend()` or manually edit your local `Project.toml` file.

!!! warning

    This step might take a while the very first time downloading all backend dependencies. 

!!! tip "Tip for CUDA" 

    CUDA.jl uses its own prebuilt CUDA stack by default, please refer to CUDA.jl docs if wanting to [use a local CUDA installation](https://cuda.juliagpu.org/stable/installation/overview/#Using-a-local-CUDA) to set up `LocalPreferences.toml`. 

!!! tip "Tip for AMDGPU"

    AMDGPU.jl relies on standard `rocm` installation under `/opt/rocm`, for non-standard locations set the environment variable `ROCM_PATH`, see [docs](https://amdgpu.juliagpu.org/stable/install_tips).

- **`JACC.@init_backend`**: initializes the selected back end automatically from `LocalPreferences.toml` from `JACC.set_backend`. `JACC.@init_backend` should be used in your code before using any JACC.jl functionality. Recent improvements have made this process more seamless.

Example:
```julia
using JACC
JACC.@init_backend
```

!!! tip "Always use `JACC.@init_backend`"

    Use `JACC.@init_backend` right after `import JACC` or `using JACC` for portable backend-agnostic code.

## Memory allocation

- **`JACC.array()`**: create a new array on the device with the specified type and size.
- **`JACC.zeros`**: create a new array on the device filled with zeros.
- **`JACC.ones`**: create a new array on the device filled with ones.
- **`JACC.fill`**: create a new array on the device filled with a specified value.
- **`JACC.to_device`**: transfer an existing Julia array from host to device.
- **`JACC.to_host`**: transfer an existing JACC array from device to host.
- **`JACC.transfer!`**: in-place version of `to_device` or `to_host`.
   
Advanced memory:
- **`JACC.Multi module`**: allows the programmability of multiple-GPU devices on a single node without the need of MPI. Please see the paper [Valero-Lara et al. IEEE eScience 2025](https://ieeexplore.ieee.org/document/11181490) 

- **`JACC.shared`**: exploits fast-access cache memory on device. Use it inside kernel functions. Please see the paper [Valero-Lara IEEE HPEC 2024](https://ieeexplore.ieee.org/document/10938453) for more details.
  
- **`JACC.@atomic`**: creates an atomic operation for safe concurrent access. Use it inside kernel functions. Wraps `@atomic` from the supported [Atomix.jl](https://github.com/JuliaConcurrent/Atomix.jl) package in the JuliaGPU ecosystem. Careful must be taken as atomic operations can be costly in terms of performance.

## Kernel launching

JACC provides two main macros/functions to launch parallel workloads on the selected back end: `JACC.@parallel_for/parallel_for` and `JACC.@parallel_reduce/parallel_reduce`.

**`JACC.@parallel_for/parallel_for`**: launch a parallel for loop (each `i` index is independent) running a "kernel" workload function with variadic arguments. 
  
- **`JACC.@parallel_reduce/parallel_reduce`**: launch a parallel reduce operation running a "kernel" workload function with variadic arguments.

!!! tip "macro vs function"

    The preferred way is to use the macros `@parallel_for` and `@parallel_reduce` as they provide a more expressive syntax separating kernel definitions (e.g., computational science) from optional launch parameters (e.g., computer science) for readability. It follows closer [Julia's rich metaprogramming philosophy from Lisp](https://docs.julialang.org/en/v1/manual/metaprogramming/).

!!! tip "`parallel_reduce` convenience functions"

    Optionally, the `parallel_reduce` function provides convenient simplified overloads for common reduction operations, e.g., `x_sum = JACC.parallel_reduce(x)` computes the sum of all elements in array `x`.


## Basic kernel launching

Basic usage of `@parallel_for/@parallel_reduce` macros implies that the user only needs to define the kernel function and then launch it with the desired range and arguments. Thus, completely abstracting away backend-specific details that are not necessarily portable.

The general format of a kernel launch is as follows:

```julia
function kernel_function(i, args...)
    # kernel code here
end

JACC.@parallel_for range=kernel_range kernel_function(args...)
result = JACC.@parallel_reduce range=kernel_range op=operator init=initial_value kernel_function(args...)
```

Example of `@parallel_for` macro:

```julia

function kernel_1D(i, args...)
    # kernel code here
end

function kernel_2D(i, j, args...)
    # kernel code here
end

function kernel_2D(i, j, k, args...)
    # kernel code here
end

JACC.@parallel_for range=N kernel_1D(args...)
JACC.@parallel_for range=(Nx, Ny) kernel_2D(args...)
JACC.@parallel_for range=(Nx, Ny, Nz) kernel_3D(args...)
```

Example of `@parallel_reduce` macro and convenience functions for sum reduction:

```julia
import JACC
JACC.@init_backend

N = 10 
x = JACC.ones(Float32, N) 

# sum reduction over array x
x_sum1 = JACC.@parallel_reduce range=N ((i,x)->x[i])(x)
@show x_sum1

function elem(i, x)
    return x[i]
end

x_sum2 = JACC.@parallel_reduce range=N elem(x)
@show x_sum2

# convenience functions for sum (default) reduction
x_sum3 = JACC.@parallel_reduce range=N JACC.elem_access(x)
@show x_sum3

x_sum4 = JACC.parallel_reduce(x)
@show x_sum4
```

Output:
```
x_sum1[] = 10.0
x_sum2[] = 10.0
x_sum3[] = 10.0
x_sum4 = 10.0f0
```

!!! warning "@parallel_reduce vs parallel_reduce" return types

    Note that the `@parallel_reduce` macro returns an array type, while the `parallel_reduce` convenience function returns the reduced value directly. Use the one that fits your needs.

In summary:
- `JACC.@parallel_for range=kernel_range kernel_function(args...)` requires a range and a kernel function with arguments.

- `JACC.@parallel_reduce range=kernel_range op=operator init=initial_value kernel_function(args...)` requires a range, an operator (+,*,min,max), an initial value, and a kernel function with arguments.

Convenience functions for common reduction operations:

- `red = JACC.parallel_reduce(a)`

- `red = JACC.parallel_reduce(op, a)` with special op= +, *, min, max

- `red = JACC.parallel_reduce(dims, dot, x1, x2)` Special op=dot product reduction, requires two arrays x1, x2.



## Advanced kernel launching

JACC also provides advanced options for kernel launching, allowing users to specify additional parameters such as blocks/thread sizes (GPU only), shared memory usage, and more. These options can be passed as keyword arguments to the `@parallel_for` and `@parallel_reduce` macros.

```
JACC.@parallel_for range=kernel_range blocks= blocks threads=threads shmem_size=shmem_size sync=true stream=stream_handler kernel_function(args...)
```

where the additional parameters are:
- `blocks`: number of blocks (GPU only)
- `threads`: number of threads per block (GPU only)
- `shmem_size`: size of shared memory to allocate in KB (GPU only)
- `stream`: stream identifier (GPU only), handler from `JACC.default_stream()` or `JACC.create_stream()` see [AMD GPU tests](https://github.com/JuliaGPU/JACC.jl/blob/main/test/backend/amdgpu.jl)
- `sync`: true or false, whether to synchronize after kernel launch (default: true)

```
JACC.@parallel_reduce range=kernel_range op=operator init=initial_value blocks=blocks threads=threads  sync=true stream=stream_handler kernel_function(args...)
```

where the additional parameters are:
- `blocks`: number of blocks (GPU only)
- `threads`: number of threads per block (GPU only)
- `stream`: stream identifier (GPU only), handler from `JACC.default_stream()` or `JACC.create_stream()` see [AMD GPU tests](https://github.com/JuliaGPU/JACC.jl/blob/main/test/backend/amdgpu.jl)
- `sync`: true or false, whether to synchronize after kernel launch (default: true)

Other examples and more advanced usages can be found in the [JACC tests directory](https://github.com/JuliaGPU/JACC.jl/blob/main/test/unittests.jl)