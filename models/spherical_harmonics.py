import sympy
import torch
from sympy import (Ynm, symbols)

from utils.sympy import SymPyModule

#---------------- Torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#------------------

class SphericalHarmonics(torch.nn.Module):
    def __init__(self, nmax=20):
        super().__init__()
        _theta = sympy.symbols("theta")
        _phi = sympy.symbols("phi")
        x, y, z = symbols("x y z", real=True)
        print("Constructing spherical harmonics functions")
        modules = []
        self.nmax = nmax
        for n in range(0, nmax + 1, 1):
            for m in range(-n, n + 1, 1):
                modules.append(
                    SymPyModule(expressions=[Ynm(n, m, _phi, _theta).expand(func=True)])
                )
        self.ylm = torch.nn.ModuleList(modules)
        self.W = torch.nn.Parameter(torch.rand(len(self.ylm) * 2))

    def forward(self, theta, phi):
        B, N = theta.shape
        res = []

        for m in self.ylm:
            e = m(phi=phi, theta=theta)
            if "complex" not in str(e.dtype):
                re, im = e, torch.zeros_like(e).to(device)
            else:
                re, im = e.real, e.imag
            if len(re.shape) < 3:
                re = re.unsqueeze(0).repeat(B, N).reshape(B, N, 1)
                im = im.unsqueeze(0).repeat(B, N).reshape(B, N, 1)
            res.append(torch.cat([re, im], -1))
        return torch.cat(res, -1)  # @sph.W
