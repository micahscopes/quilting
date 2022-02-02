from alglbraic.algebras.clifford_algebra import ConformalGeometricAlgebra, CliffordAlgebra
from sympy import diag, Matrix, sqrt, symbols, Symbol, eye, Matrix
from sympy.tensor import IndexedBase

from alglbraic.glsl import GLSL
from alglbraic.functions import map


class CGA(ConformalGeometricAlgebra):
    def __init__(self, size, name=None, **opts):
        name = name if name else ("CGA%i" % size)
        grade_1_basis_names = ["e%i" % (i + 1) for i in range(size)] + [
            "nil",
            "inf",
        ] + ["dual1", "dual2"]

        quadratic_form = diag(eye(size), Matrix([[0, 1], [1, 0]]), 0, 0)

        CliffordAlgebra.__init__(
            self, name, quadratic_form, grade_1_basis_names, **opts
        )

        self.nil, self.inf = self._grade_1_basis[-4:-2]
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

    # @GLSL
    # def point_coords(self):
    #     nil, inf = self.nil, self.inf
    #     u = self.algebraic_arguments(1)
    #     normalize = lambda pt: pt / (pt | inf).scalar()
    #     result = normalize(u)
    #     # result = (-u/(u.grade(2)|self.inf)^self.mnk)/self.mnk

    #     return self.algebraic_operation("point_coords", result, n=1)
