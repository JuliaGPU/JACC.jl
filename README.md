![](./docs/src/assets/logo.png)

# JACC.jl: Julia for Accelerators

[![Documentation](https://github.com/JuliaGPU/JACC.jl/actions/workflows/documentation.yaml/badge.svg)](https://juliagpu.github.io/JACC.jl)
[![ci-cpu](https://github.com/JuliaGPU/JACC.jl/actions/workflows/ci-cpu.yaml/badge.svg)](https://github.com/JuliaGPU/JACC.jl/actions/workflows/ci-cpu.yaml)
[![ci-gpu-NVIDIA](https://github.com/JuliaGPU/JACC.jl/actions/workflows/ci-gpu-NVIDIA.yaml/badge.svg)](https://github.com/JuliaGPU/JACC.jl/actions/workflows/ci-gpu-NVIDIA.yaml)
[![ci-gpu-AMD](https://github.com/JuliaGPU/JACC.jl/actions/workflows/ci-gpu-AMD.yaml/badge.svg)](https://github.com/JuliaGPU/JACC.jl/actions/workflows/ci-gpu-AMD.yaml)
[![ci-gpu-Apple](https://github.com/JuliaGPU/JACC.jl/actions/workflows/ci-gpu-Apple.yaml/badge.svg)](https://github.com/JuliaGPU/JACC.jl/actions/workflows/ci-gpu-Apple.yaml)
[![codecov](https://codecov.io/github/JuliaGPU/JACC.jl/graph/badge.svg?token=u4Td0r52ng)](https://codecov.io/github/JuliaGPU/JACC.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub release](https://img.shields.io/github/release/JuliaGPU/JACC.jl/all.svg)](https://github.com/JuliaGPU/JACC.jl/releases)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12117/badge)](https://www.bestpractices.dev/projects/12117)


<img src="./docs/src/assets/jacc-metal.gif" width="95%" height="650"/>

*JACC.jl on Apple M3 GPU* [source](./docs/src/assets/jacc-replay.jl)


CPU/GPU portable `array`/`parallel_for`/`parallel_reduce` in Julia for productive science.

JACC.jl leverages the LLVM-based Julia language and ecosystem, in particular [JuliaGPU](https://juliagpu.org/), and [optional package
extensions](https://julialang.org/blog/2023/04/julia-1.9-highlights/#package_extensions). Similar to portable layers like Kokkos, RAJA, or SYCL in C++, Julia users will have easy access to vendor-neutral CPU/GPU computing writing a single 100% portable high-level source code, but in a language like Julia designed for science. 

JACC.jl programming model provides:
  
  1. Portable `array`, `zeros`, `ones`, `fill` metaprogramming for the selected vendor backend (`CuArray`. `ROCArray`, `MtlArray`, etc.).

  2. `parallel_for` and `parallel_reduce` kernel launching: (i) basic APIs for non-experts, and (ii) low-level control APIs for threads/blocks, synchronization, multi GPU, and shared memory usage.

  3. Backend selection using Preferences.jl: `"threads"` (default), `"cuda"`, `"amdgpu"`, `"metal"` and `"oneAPI"`. Stored in Julia's `LocalPreferences.toml`, so code is 100% vendor-agnostic via `@init_backend`.

## Goals
  1. JACC.jl provides easy access to vendor-neutral parallel computing in Julia without having to learn the vendor-specific details of each backend or CPU/GPU parallel programming or extra code annotations. Developers can write CPU/GPU kernels interactively and deploy on multiple high-performance computing (HPC) platforms.

  2. Julia HPC developers can use JACC.jl as a productive meta-programming layer that adds and communicates use-case and testing coverage to the ever-growing JuliaGPU ecosystem.

  3. As a platform to advance research in portable parallel programming systems, e.g. [shared memory](https://ieeexplore.ieee.org/document/10938453), [multi-GPU](https://ieeexplore.ieee.org/document/11181490), [for science experimental facilities](https://ieeexplore.ieee.org/document/10820586), etc.

## Support and Roadmap

Julia provides a tight, interoperable ecosystem for GPU programming. Still, vendor support of some features may vary. The following table summarizes the current support status of JACC.jl features across different backends.


| Feature \ Backend | CPU                 | CUDA              | AMDGPU | Metal | oneAPI            |
| ----------------- | ------------------- | ----------------- | ------ | ----- | ----------------- |
| CI                | ✅                   | ✅                 | ✅      | ✅     | TBD               |
|                   | x86, Arm GH Runners | RTXA4000, GTX1080 | MI100  | M1    | TBD               |
| Float64           | ✅                   | ✅                 | ✅      | ❌     | ✅  (if supported) |
| `Multi` (GPU)     | N/A                 | ✅                 | ❌      | ❌     | ❌                 |
| `shared`          | N/A                 | ✅                 | ✅      | ✅     | ✅                 |
| `@atomic`         | ✅                   | ✅                 | ✅      | ✅     | ✅                 |

Roadmap:

- JACC.BLAS for kernel-level linear algebra routines.
- Expand JACC's ecosystem including scientific applications, see [JACC-Applications](https://github.com/JuliaORNL/JACC-applications)
- New functionality impacting scientific users.

## Quick start

1. JACC.jl is a registered Julia package. Install JACC.jl using the Julia package manager:

    ```julia
    $ julia
    julia> import Pkg
    julia> Pkg.add("JACC")
    ```
    or 
    
    ```julia
    julia> using JACC
    julia> ] 
    (tmp) pkg> add JACC
    ```

2. Set a backend (outside code): `"cuda"`, `"amdgpu"`, `"metal"`, `"oneapi"`, or `"threads"` (default). This will generate a `LocalPreferences.toml` file.

    ```julia
    julia> import JACC
    julia> JACC.set_backend("cuda")
    ```
    **NOTE:** This will also add the backend package (`CUDA.jl` in this case)
    as a dependency to the current project in Project.toml. Users can clean this using `unset_backend`.

3. Import JACC and load appropriate extension. `@init_backend` inserts an `import` statement so that you don't have to reference a specific backend in your code. (It must therefore be called at a top-level scope.)

    ```julia
    import JACC
    JACC.@init_backend
    ```

    **NOTE:** Without calling `@init_backend` if your backend is something other
    than `"threads"`, using most of JACC functions will result in an error
    related to, using CUDA for example, `get_backend(::Val(:cuda))`.

4. Run a kernel example (see tests directory), copy the code below into a `jacc-example.jl` file:

    ```julia
    import JACC
    JACC.@init_backend

    function axpy(i, alpha, x, y)
      @inbounds x[i] += alpha * y[i]
    end

    N = 100_000
    alpha = Float32(2.0)
    x = JACC.zeros(Float32, N)
    y = JACC.array(fill(Float32(5), N))
    JACC.@parallel_for range=N axpy(alpha, x, y)
    sum_x = JACC.@parallel_reduce range=N ((i,x)->x[i])(x)
    println("Result: ", sum_x)
    ```

    and run it:

    - CPU using 4 threads: `$ julia -t 4 jacc-example.jl` 
    - GPU: `$ julia jacc-example.jl` 
    
    The first time it will take a while to download and precompile backend dependency packages.

    **NOTE:** `JACC.array` converts a `Base.Array` to the array type used by the current backend. If you need to use the array type in your code (for example as function parameter or in a struct, use `JACC.array_type()`:
    ```julia
    const JACCArray = JACC.array_type()
    function f(a::JACCArray{Float32, 1})
        # ...
    end
    ```

## Resources

- Documentation: [https://juliagpu.github.io/JACC.jl](https://juliagpu.github.io/JACC.jl)
- For an app integration example see the [GrayScott.jl](https://github.com/JuliaORNL/GrayScott.jl) and the
  [Simulation.jl code](https://github.com/JuliaORNL/GrayScott.jl/blob/main/src/simulation/Simulation.jl) and search for the `JACC` keyword.
- JuliaCon 2023 presentation [video](https://live.juliacon.org/talk/AY8EUX).
- SC24 JACC [Best Research Poster Finalist](https://sc24.supercomputing.org/proceedings/poster/poster_pages/post113.html)
- SC24 WACCPD [presentation](https://sc24.conference-program.com/presentation/?id=ws_waccpd101&sess=sess760)
  and [paper](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400b955/555400b955.pdf)
- MiniVATES.jl proxy application [repository](https://github.com/JuliaORNL/MiniVATES.jl)
  and SC24 XLOOP [best paper using JACC.jl](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400c107/555400c107.pdf)
- [OLCF Tutorial 2025](https://www.olcf.ornl.gov/calendar/juliaforsci2025/)
- Examples of [science kernels using JACC.jl](https://github.com/JuliaORNL/JACC-applications)

## Citation

If you find JACC.jl useful please cite the following paper from [SC24 WACCPD](https://doi.org/10.1109/SCW63240.2024.00245), open version available [here](https://conferences.computer.org/sc-wpub/pdfs/SC-W2024-6oZmigAQfgJ1GhPL0yE3pS/555400b955/555400b955.pdf).

```
@INPROCEEDINGS{JACC,
  author={Valero-Lara, Pedro and Godoy, William F and Mankad, Het and Teranishi, Keita and Vetter, Jeffrey S and Blaschke, Johannes and Schanen, Michel},
  booktitle={Proceedings of the SC '24 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis},
  title={{JACC: Leveraging HPC Meta-Programming and Performance Portability with the Just-in-Time and LLVM-based Julia Language}},
  year={2024},
  volume={},
  number={},
  pages={},
  doi={10.1109/SCW63240.2024.00245}
}
```

## Sponsors:

JACC.jl is funded by the US Department of Energy Advanced Scientific Computing Research (ASCR) projects:

- [S4PST](https://s4pst.org/), part of the Next Generation of Scientific Software Technologies (NGSST) ASCR Program. 
- NGSST sponsors the Consortium for the Advancement of Scientific Software, [CASS](https://cass.community/)
- ASCR Competitive Portfolios for Advanced Scientific Computing Research, MAGMA/Fairbanks project

Former sponsors:

- [ASCR Bluestone X-Stack](https://csmd.ornl.gov/Bluestone)
- The Exascale Computing Project - [PROTEAS-TUNE](https://www.ornl.gov/project/proteas-tune) 
