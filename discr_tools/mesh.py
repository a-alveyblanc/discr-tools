from meshmode.mesh import TensorProductElementGroup
from meshmode.mesh.generation import generate_regular_rect_mesh


class Mesh:
    """
    Used to generate and store a uniform quadrilateral mesh
    """
    def __init__(self, a, b, dim, nelts_1d):
        self.a = a
        self.b = b
        self.dim = dim
        self.nelts_1d = nelts_1d

        self._mesh = generate_regular_rect_mesh(
            (self.a,)*self.dim, (self.b,)*self.dim,
            nelements_per_axis=(self.nelts_1d,)*self.dim,
            group_cls=TensorProductElementGroup
        )

        self._vertices = self.mesh.vertices
        self._vertex_idxs = self.mesh.groups[0].vertex_indices
        self._elements = self.vertices[:,self._vertex_idxs]

    @property
    def mesh(self):
        return self._mesh

    @property
    def elements(self):
        return self._elements

    @property
    def vertices(self):
        return self._vertices

    @property
    def vertex_indices(self):
        return self._vertex_idxs
