# discr-tools

Abstraction for solving PDEs using SEM/FEM.
- (Some) GPU support via https://github.com/inducer/loopy
    - See `kernels.py` for a list of kernels explicitly utilizing a GPU
- Meshing via https://github.com/inducer/meshmode
- TODO: algebraic abstraction similar to `libCEED` for applying operators in an
  element-local way without needing to explicitly perform gather-scatter
  operations
