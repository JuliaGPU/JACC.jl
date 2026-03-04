
# Contributing to JACC.jl

We welcome contributions to JACC.jl! Whether it's fixing a bug, adding a new feature, improving documentation, or enhancing tests, your help is appreciated. Here are some guidelines to get you started:

JACC.jl aims to provide a performance portable and productive programming model for CPU/GPU execution within the Julia ecosystem. We welcome contributions that improve performance, portability, usability, documentation, and testing across architectures.

---

## Table of Contents

- [Contributing to JACC.jl](#contributing-to-jaccjl)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Ways to Contribute](#ways-to-contribute)
  - [Development Setup](#development-setup)
    - [1. Fork and Clone](#1-fork-and-clone)
    - [2. Install Julia](#2-install-julia)
    - [3. Instantiate the Environment](#3-instantiate-the-environment)
    - [4. Set Backend Environment Variables (if applicable)](#4-set-backend-environment-variables-if-applicable)
  - [Running Tests](#running-tests)
  - [GPU Testing](#gpu-testing)
  - [Code Style Guidelines](#code-style-guidelines)
  - [Performance Contributions](#performance-contributions)
  - [Documentation](#documentation)
  - [Submitting a Pull Request](#submitting-a-pull-request)
  - [Reporting Issues](#reporting-issues)
  - [Roadmap and Discussions](#roadmap-and-discussions)
  - [Thank You](#thank-you)

---

## Code of Conduct

This project follows the [Julia Community Standards](https://julialang.org/community/standards/).

Please be respectful and constructive in all discussions and reviews.

---

## Ways to Contribute

We welcome:

* 🐛 Bug reports
* 💡 Feature requests for use cases
* ⚡ Performance improvements (CPU and GPU)
* 🧪 Additional test coverage
* 📖 Documentation improvements
* 🧵 Backend support improvements (CUDA.jl, AMDGPU.jl, Metal.jl, oneAPI.jl)
* 📊 Benchmarks and reproducible performance cases

If you are unsure whether something fits the project scope, feel free to open an issue first.

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/JACC.jl.git
cd JACC.jl
```

### 2. Install Julia

Install a recent stable version of Julia:

👉 [https://julialang.org/downloads/](https://julialang.org/downloads/)

### 3. Instantiate the Environment

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

### 4. Set Backend Environment Variables (if applicable)

```bash
julia --project
```

Inside Julia:

```julia
using JACC
JACC.set_backend(JACC.Backend.cuda)  # or amdgpu, metal, oneapi, threads, etc.
```

---

## Running Tests

Run the full test suite:

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

Please ensure:

* All tests pass locally before submitting a PR.
* New functionality includes appropriate tests.
* Edge cases are covered as much as possible.

---

## GPU Testing

JACC.jl targets portable CPU/GPU execution.

If contributing GPU-related changes:

* Test on at least one supported backend (e.g., CUDA.jl or AMDGPU.jl).
* Clearly state which backend(s) you tested.
* Provide performance numbers if the change affects execution behavior if applicable.

If a GPU is unavailable, ensure CPU execution remains correct and include a note in the PR describing limitations.
We use GPU systems on CI, but local testing is crucial for development.

---

## Code Style Guidelines

We follow standard Julia conventions:

* Follow the official Julia Style Guide
  [https://docs.julialang.org/en/v1/manual/style-guide/](https://docs.julialang.org/en/v1/manual/style-guide/)
* Use descriptive function and variable names.
* Add docstrings to all public APIs.
* Keep kernels and execution abstractions clean and minimal.
* Avoid unnecessary type annotations unless required for dispatch or performance.

If formatting is used in the repository:

```bash
julia --project -e 'using JuliaFormatter; format(".")'
```

it will apply .JuliaFormatter.toml settings to the entire codebase. Please run it before submitting a PR to ensure consistent formatting. VS Code users can install the Julia extension and enable format on save for convenience, this requires JuliaFormatter to be installed for your Julia project environment or globally (e.g., `import Pkg; Pkg.add("JuliaFormatter")`).

---

## Performance Contributions

Performance is central to JACC.jl.

If your PR impacts performance:

* Provide benchmarks results.
* Include system information (CPU/GPU model, Julia version).
* Compare before vs. after results.
* Avoid micro-optimizations without measurable benefit.

For large performance changes, consider attaching a reproducible benchmark script.

---

## Documentation

* Use triple-quoted docstrings `"""`.
* Include minimal examples in docstrings.
* Keep examples portable across CPU/GPU when possible.
* If documentation is built with Documenter.jl, ensure it builds locally.

To build documentation (if applicable):

```bash
julia --project=docs docs/make.jl
```

Our documentation is also tested on CI, but our online [docs](https://juliagpu.github.io/JACC.jl) won't update until a PR is merged. If your PR includes documentation changes, please ensure they are correct and build successfully locally.

---

## Submitting a Pull Request

1. Create a feature branch:

   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes.

3. Ensure tests pass.

4. Push to your fork.

5. Open a Pull Request against `main`.

Please:

* Clearly describe what the PR does.
* Reference related issues (e.g., `Fixes #123`).
* Keep changes focused and atomic.
* Be responsive to review feedback.

Draft PRs are welcome for early feedback.

---

## Reporting Issues

When opening an issue, please include:

* Julia version (`versioninfo()`)
* Backend used (CUDA.jl, AMDGPU.jl, Metal.jl, oneAPI.jl, etc.)
* Hardware information (GPU model if applicable)
* Operating system
* Minimal working example
* Expected vs. actual behavior

Performance issues should include benchmark details.

---

## Roadmap and Discussions

For larger design discussions:

* Open a GitHub Discussion (if enabled)
* Propose a design sketch in an issue
* Start with a draft PR for feedback

---

## Thank You

Your contributions help make JACC.jl a robust and portable solution for high-performance CPU/GPU programming in Julia.

We appreciate your time and expertise! 🚀

Maintainers are funded by the US Department of Energy, Advanced Scientific Computing Research (ASCR) program. Please be aware that we will prioritize contributions that align with our mission of advancing scientific computing and performance portability. Contributions that enhance the capabilities of JACC.jl for scientific applications, improve performance on DOE-relevant hardware, or expand support for emerging architectures are especially valuable to us.

Special thanks to the [JuliaGPU](https://www.juliagpu.org) community for their ongoing support and contributions to GPU programming in Julia. Your work is instrumental in making JACC.jl a success! 🙌