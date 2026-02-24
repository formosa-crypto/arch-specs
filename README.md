# arch-specs

[![checkspecs CI](https://github.com/formosa-crypto/arch-specs/actions/workflows/checkspecs.yml/badge.svg)](https://github.com/formosa-crypto/arch-specs/actions/workflows/checkspecs.yml)

Repository for architecture-level specification checks, centered on validating
AVX2 specs against reference implementations / hardware.

## Repository Layout

- `checkspecs/`: OCaml/Dune project containing the `checkspecs` executable.
- `specs/`: current specification files to validate (for example `specs/avx2.spec`).

## Local Build and Run

From repository root:

```bash
cd checkspecs
opam install . --deps-only -y
opam exec -- dune build src/checkspecs.exe
opam exec -- dune exec checkspecs -- -n 10000 ../specs/avx2.spec
```

Optional filtering by instruction name:

```bash
opam exec -- dune exec checkspecs -- -n 10000 --filter VPAND ../specs/avx2.spec
```
