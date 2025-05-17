# discr-tools

Abstraction for solving PDEs using SEM/FEM.
- (Some) GPU support via https://github.com/inducer/loopy
    - See `kernels.py` for a list of kernels explicitly utilizing a GPU
- Meshing via https://github.com/inducer/meshmode
- `Matvec` class applies gather-scatter operations allowing users to define matrix actions.
     - Useful for, e.g., solving matrix-free with iterative methods
