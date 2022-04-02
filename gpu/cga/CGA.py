from alglbraic.algebras.clifford_algebra import ConformalGeometricAlgebra
from sympy import diag, Matrix, sqrt, symbols, Symbol
from sympy.tensor import IndexedBase

from alglbraic.glsl import GLSL
from alglbraic.functions import map

class CGA(ConformalGeometricAlgebra):
    def __init__(self, size, **opts):
        ConformalGeometricAlgebra.__init__(self, size, **opts)
        self.nil, self.inf = self._grade_1_basis[-2:]
        self.mnk = self.nil ^ self.inf

    @GLSL
    def INF(self):
        result = self.inf
        return self.algebraic_operation("INF", result, n=0)

    @GLSL
    def NIL(self):
        result = self.nil
        return self.algebraic_operation("NIL", result, n=0)

    @GLSL
    def MNK(self):
        result = self.mnk
        return self.algebraic_operation("MNK", result, n=0)

    @GLSL
    def point(self):
        nil, inf = self.nil, self.inf
        Pt = lambda x: (x | x) * inf / 2 + x + nil
        u = self.algebraic_arguments(1)
        result = Pt(u)
        return self.algebraic_operation("point", result, n=1)

    @GLSL
    def point_coords(self):
        nil, inf = self.nil, self.inf
        u = self.algebraic_arguments(1)
        normalize = lambda pt: pt / (pt | inf).scalar()
        result = normalize(u)
        # result = (-u/(u.grade(2)|self.inf)^self.mnk)/self.mnk

        return self.algebraic_operation("point_coords", result, n=1)

    # @GLSL
    # def dual_sphere(self):
    #     nil, inf = self.nil, self.inf
    #     Sph = lambda x, rad: ((x | x) - rad ** 2) * inf / 2 + x + nil
    #     center = self.algebraic_arguments(1)
    #     radius = Symbol("radius")
    #     result = Sph(center, radius)

    #     input_types = [self.type_name, self.base_ring]
    #     input_argnames = [self.U, "radius"]

    #     return map(
    #         "dual_sphere",
    #         input_types,
    #         input_argnames,
    #         self.type_name,
    #         self._coefficients_from_algebraic_element(result),
    #     )