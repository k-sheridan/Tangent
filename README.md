<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/tangent-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/_static/tangent-logo-light.svg">
    <img src="docs/_static/tangent-logo-light.svg" alt="Tangent Logo" width="300">
  </picture>
</p>

# Tangent

[![Tests](https://github.com/k-sheridan/Tangent/actions/workflows/test.yml/badge.svg)](https://github.com/k-sheridan/Tangent/actions/workflows/test.yml)
[![Benchmarks](https://github.com/k-sheridan/Tangent/actions/workflows/benchmark.yml/badge.svg)](https://github.com/k-sheridan/Tangent/actions/workflows/benchmark.yml)
[![Documentation](https://github.com/k-sheridan/Tangent/actions/workflows/docs.yml/badge.svg)](https://k-sheridan.github.io/Tangent/)

**Header-only generic optimizer for manifold-based nonlinear least squares**

## Features

- SE3/SO3 manifold optimization with Lie algebra
- Automatic differentiation (no manual Jacobians needed)
- Sparse Schur complement solver
- Marginalization support via Sparse Gaussian Prior
- Cache-friendly SlotMap containers (O(1) operations)
- Compile-time type safety

## Documentation

### [Getting Started](https://k-sheridan.github.io/Tangent/getting-started.html)

Installation, requirements, and a quick start guide with code examples.

### [Concepts](https://k-sheridan.github.io/Tangent/concepts/index.html)

Deep dives into the core building blocks:

- **Variables** — Defining optimizable parameters on manifolds
- **Error Terms** — Creating constraints between variables
- **Autodiff** — How automatic differentiation works

### [API Reference](https://k-sheridan.github.io/Tangent/api/index.html)

Complete class and function documentation.

## Origin

Tangent was originally developed as part of [QDVO (Quasi-Direct Visual Odometry)](https://github.com/k-sheridan/qdvo).
