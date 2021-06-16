import sympy, torch

import collections as co
import functools as ft
import sympy
import torch
from sympy import (sympify, factorial, var, cos, S, sin, Dummy, sqrt, pi, exp,
        I, latex, symbols)

def Plm(l, m, z):
    """
    Returns the associated Legendre polynomial P_{lm}(z).
    The Condon & Shortley (-1)^m factor is included.
    """
    l = sympify(l)
    m = sympify(m)
    z = sympify(z)
    if m >= 0:
        r = ((z**2-1)**l).diff(z, l+m)
        return (-1)**m * (1-z**2)**(m/2) * r / (2**l * factorial(l))
    else:
        m = -m
        r = ((z**2-1)**l).diff(z, l+m)
        return factorial(l-m)/factorial(l+m) * (1-z**2)**(m/2) * r / (2**l * factorial(l))


def Plm_cos(l, m, theta):
    """
    Returns the associated Legendre polynomial P_{lm}(cos(theta)).
    The Condon & Shortley (-1)^m factor is included.
    """
    l = sympify(l)
    m = sympify(m)
    theta = sympify(theta)
    z = Dummy("z")
    r = ((z**2-1)**l).diff(z, l+m).subs(z**2-1, -sin(theta)**2).subs(z, cos(theta))
    return (-1)**m * sin(theta)**m * r / (2**l * factorial(l))


def Ylm(l, m, theta, phi):
    """
    Returns the spherical harmonics Y_{lm}(theta, phi) using the Condon & Shortley convention.
    """
    l, m, theta, phi = sympify(l, rational=False), sympify(m, rational=False), sympify(theta, rational=False), sympify(phi, rational=False)
    return sqrt((2*l+1)/(4*pi) * factorial(l-m)/factorial(l+m)) * Plm_cos(l, m, theta) * exp(I*m*phi)


def Zlm(l, m, theta, phi):
    """
    Returns the real spherical harmonics Z_{lm}(theta, phi).
    """
    l, m, theta, phi = sympify(l), sympify(m), sympify(theta), sympify(phi)
    if m > 0:
        return sqrt((2*l+1)/(2*pi) * factorial(l-m)/factorial(l+m)) * Plm_cos(l, m, theta) * cos(m*phi)
    elif m < 0:
        m = -m
        return sqrt((2*l+1)/(2*pi) * factorial(l-m)/factorial(l+m)) * Plm_cos(l, m, theta) * sin(m*phi)
    elif m == 0:
        return sqrt((2*l+1)/(4*pi)) * Plm_cos(l, 0, theta)
    else:
        raise ValueError("Invalid m.")

def Zlm_xyz(l, m, x, y, z):
    """
    Returns the real spherical harmonics Z_{lm}(x, y, z).
    It is assumed x**2 + y**2 + z**2 == 1.
    """
    l, m, x, y, z = sympify(l), sympify(m), sympify(x), sympify(y), sympify(z)
    if m > 0:
        r = (x+I*y)**m
        r = r.as_real_imag()[0]
        return sqrt((2*l+1)/(2*pi) * factorial(l-m)/factorial(l+m)) * Plm(l, m, z) * r / sqrt(1-z**2)**m
    elif m < 0:
        m = -m
        r = (x+I*y)**m
        r = r.as_real_imag()[1]
        return sqrt((2*l+1)/(2*pi) * factorial(l-m)/factorial(l+m)) * Plm(l, m, z) * r / sqrt(1-z**2)**m
    elif m == 0:
        return sqrt((2*l+1)/(4*pi)) * Plm(l, 0, z)
    else:
        raise ValueError("Invalid m.")


def _reduce(fn):
    def fn_(*args):
        return ft.reduce(fn, args)
    return fn_


_global_func_lookup = {
    sympy.Mul: _reduce(torch.mul),
    sympy.Add: _reduce(torch.add),
    sympy.div: torch.div,
    sympy.Abs: torch.abs,
    sympy.sign: torch.sign,
    # Note: May raise error for ints.
    sympy.ceiling: torch.ceil,
    sympy.floor: torch.floor,
    sympy.log: torch.log,
    sympy.exp: torch.exp,
    sympy.sqrt: torch.sqrt,
    sympy.cos: torch.cos,
    sympy.acos: torch.acos,
    sympy.sin: torch.sin,
    sympy.asin: torch.asin,
    sympy.tan: torch.tan,
    sympy.atan: torch.atan,
    sympy.atan2: torch.atan2,
    # Note: May give NaN for complex results.
    sympy.cosh: torch.cosh,
    sympy.acosh: torch.acosh,
    sympy.sinh: torch.sinh,
    sympy.asinh: torch.asinh,
    sympy.tanh: torch.tanh,
    sympy.atanh: torch.atanh,
    sympy.Pow: torch.pow,
    sympy.re: torch.real,
    sympy.im: torch.imag,
    sympy.arg: torch.angle,
    # Note: May raise error for ints and complexes
    sympy.erf: torch.erf,
    sympy.loggamma: torch.lgamma,
    sympy.Eq: torch.eq,
    sympy.Ne: torch.ne,
    sympy.StrictGreaterThan: torch.gt,
    sympy.StrictLessThan: torch.lt,
    sympy.LessThan: torch.le,
    sympy.GreaterThan: torch.ge,
    sympy.And: torch.logical_and,
    sympy.Or: torch.logical_or,
    sympy.Not: torch.logical_not,
    sympy.Max: torch.max,
    sympy.Min: torch.min,
    # Matrices
    sympy.MatAdd: torch.add,
    sympy.HadamardProduct: torch.mul,
    sympy.Trace: torch.trace,
    # Note: May raise error for integer matrices.
    sympy.Determinant: torch.det,
    sympy.core.numbers.ImaginaryUnit: lambda *args: torch.complex(torch.Tensor([0]), torch.Tensor([1])).cuda()
}

class _Node(torch.nn.Module):
    def __init__(self, *, expr, _memodict, _func_lookup, **kwargs):
        super().__init__(**kwargs)

        self._sympy_func = expr.func
        #print(expr, expr.func, expr.args)
        try:
          expr = sympy.Float(expr)
          #print(expr, expr.func, expr.args, sympy.Float(expr))
        except:
          pass
        if issubclass(expr.func, sympy.Float):
            self._value = torch.nn.Parameter(torch.tensor(float(expr)))
            self._torch_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.UnevaluatedExpr):
            if len(expr.args) != 1 or not issubclass(expr.args[0].func, sympy.Float):
                raise ValueError("UnevaluatedExpr should only be used to wrap floats.")
            self.register_buffer('_value', torch.tensor(float(expr.args[0])))
            self._torch_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.Integer):
            # Can get here if expr is one of the Integer special cases,
            # e.g. NegativeOne
            self._value = int(expr)
            self._torch_func = lambda: self._value
            self._args = ()
        elif issubclass(expr.func, sympy.Symbol):
            self._name = expr.name
            self._torch_func = lambda value: value
            self._args = ((lambda memodict: memodict[expr.name]),)
        else:
            self._torch_func = _func_lookup[expr.func]
            args = []
            for arg in expr.args:
                try:
                    arg_ = _memodict[arg]
                except KeyError:
                    arg_ = type(self)(expr=arg, _memodict=_memodict, _func_lookup=_func_lookup, **kwargs)
                    _memodict[arg] = arg_
                args.append(arg_)
            self._args = torch.nn.ModuleList(args)

    def sympy(self, _memodict):
        if issubclass(self._sympy_func, sympy.Float):
            return self._sympy_func(self._value.item())
        elif issubclass(self._sympy_func, sympy.UnevaluatedExpr):
            return self._sympy_func(self._value.item())
        elif issubclass(self._sympy_func, sympy.Integer):
            return self._sympy_func(self._value)
        elif issubclass(self._sympy_func, sympy.Symbol):
            return self._sympy_func(self._name)
        else:
            args = []
            for arg in self._args:
                try:
                    arg_ = _memodict[arg]
                except KeyError:
                    arg_ = arg.sympy(_memodict)
                    _memodict[arg] = arg_
                args.append(arg_)
            return self._sympy_func(*args)

    def forward(self, memodict):
        args = []
        for arg in self._args:
            try:
                arg_ = memodict[arg]
            except KeyError:
                arg_ = arg(memodict)
                memodict[arg] = arg_
            args.append(arg_)
        return self._torch_func(*args)


class SymPyModule(torch.nn.Module):
    def __init__(self, *, expressions, extra_funcs=None, **kwargs):
        super().__init__(**kwargs)

        if extra_funcs is None:
            extra_funcs = {}
        _func_lookup = co.ChainMap(_global_func_lookup, extra_funcs)

        _memodict = {}
        self._nodes = torch.nn.ModuleList(
            [_Node(expr=expr, _memodict=_memodict, _func_lookup=_func_lookup) for expr in expressions]
        )

    def sympy(self):
        _memodict = {}
        return [node.sympy(_memodict) for node in self._nodes]

    def forward(self, **symbols):
        return torch.stack([node(symbols) for node in self._nodes], dim=-1)


class SymModule(torch.nn.Module):
    def __init__(self, sym):
        super().__init__()
        self.sym = SymPyModule(expressions=[sym])

    def forward(self, phi, theta):
      res = self.sym(theta=theta, phi=phi)
      if 'complex' not in str(res.dtype):
        return res, torch.zeros_like(res).cuda()
      else:
        return res.real, res.imag