import sympy as sympy
import numpy as np
exp = sympy.exp
sqrt = sympy.sqrt


class MathFunction:
    def __init__(self, n):
        self.xs = self.get_xs(n)
        self.expr = sympy.simplify(self.get_expr())
        self.derivatives = self.get_derivatives()
        self.hessian = self.get_hessian()

    def get_xs(self, n):
        return [sympy.Symbol('x'+str(i+1)) for i in range(n)]

    def get_expr(self):
        raise NotImplementedError

    @staticmethod
    def get_expr_from_rs(rs, m):
        expr = 0
        for i in range(m):
            expr+=rs[i]**2
        return sympy.simplify(expr)

    def get_derivatives(self):
        derivatives = []
        for x in self.xs:
            derivatives.append(sympy.diff(self.expr, x))
        return derivatives

    def get_hessian(self):
        hessian = []
        for derivative in self.derivatives:
            hessian.append([])
            for x in self.xs:
                hessian[-1].append(sympy.diff(derivative, x))
        return hessian

    def calculate_value(self, input_xs_):
        subs = {}
        for i in range(self.n):
            subs[self.xs[i]]= input_xs_[i]
        return np.float64(self.expr.evalf(100, subs=subs))

    def calculate_derivative(self, input_xs_):
        subs = {}
        for i in range(self.n):
            subs[self.xs[i]] = input_xs_[i]
        res = []
        for derivative in self.derivatives:
            res.append(np.float64(derivative.evalf(100, subs=subs)))
        return np.array(res, np.float64)

    def calculate_hessian(self,input_xs_):
        subs = {}
        for i in range(self.n):
            subs[self.xs[i]] = input_xs_[i]
        res = []
        for exprs in self.hessian:
            res.append([])
            for expr in exprs:
                res[-1].append(np.float64(expr.evalf(100, subs=subs)))
        return np.array(res, np.float64)

class PDQ(MathFunction):
    def __init__(self, n):
        self.n = n
        self.m = n
        MathFunction.__init__(self, self.n)

    def get_expr(self):
        rs = []
        xs = self.xs
        for i in range(self.n):
            rs.append(xs[i])
        return self.get_expr_from_rs(rs, self.m)

class BoxThreeDimensional(MathFunction):
    def __init__(self, m=3):
        self.n=3
        self.m=m
        MathFunction.__init__(self, self.n)
    def get_expr(self):
        rs = []
        xs = self.xs
        for i in range(1, self.m+1):
            ti = 0.1*i
            rs.append(
                exp(-ti*xs[0])-exp(-ti*xs[1])-xs[2]*(exp(-ti)-exp(-10*ti))
            )
        return self.get_expr_from_rs(rs, self.m)

class Penalty2(MathFunction):
    def __init__(self, n, alpha=1e-5):
        self.n = n
        self.m = 2*n
        self.alpha=alpha
        MathFunction.__init__(self, self.n)
    def get_expr(self):
        xs = self.xs
        #r1
        rs = [xs[0]-0.2]
        #r2->rn
        for i in range(2, self.n+1):
            yi = exp(i/10)+exp((i-1)/10)
            rs.append(sqrt(self.alpha)*(exp(xs[i-1]/10)+exp(xs[i-2]/10)-yi))
        for i in range(self.n+1, 2*self.n):
            rs.append(sqrt(self.alpha)*(exp(xs[i-self.n+1-1])-exp(-1/10)))
        #r2n
        r2n = 0
        for j in range(1, self.n+1):
            r2n += (self.n-j+1)*xs[j-1]**2
        r2n -= 1
        rs.append(r2n)
        return self.get_expr_from_rs(rs, self.m)


# class Penalty1(MathFunction):
#     def __init__(self, n, gamma=1e-5):
#         self.n = n
#         self.m = n+1
#         self.gamma = gamma
#         MathFunction.__init__(self, self.n)
#
#     def get_expr(self):
#         rs = [sympy.sqrt(self.gamma)*(self.xs[i]-1) for i in range(self.n)]
#         r_n_plus_1 = 0
#         for x in self.xs:
#             r_n_plus_1 += (self.n * x * x)
#         r_n_plus_1 -= sympy.Rational(1, 4)
#         rs.append(r_n_plus_1)
#         return self.get_expr_from_rs(rs, self.m)
#
# class Rosenbrock(MathFunction):
#     def __init__(self):
#         self.n=2
#         self.m=2
#         MathFunction.__init__(self, self.n)
#     def get_expr(self):
#         rs = [10*(self.xs[1]-self.xs[0]**2), 1-self.xs[0]]
#         return self.get_expr_from_rs(rs, self.m)
#
# class PowellBadlyScaled(MathFunction):
#     def __init__(self):
#         self.n=2
#         self.m=2
#         MathFunction.__init__(self, self.n)
#     def get_expr(self):
#         rs = [1e4*self.xs[0]*self.xs[1]-1, exp(-self.xs[0])+exp(-self.xs[1])-1-(1e-4)]
#         return self.get_expr_from_rs(rs,self.m)
#
# class Wood(MathFunction):
#     def __init__(self):
#         self.n = 4
#         self.m = 6
#         MathFunction.__init__(self, self.n)
#     def get_expr(self):
#         rs = [
#             10*(self.xs[1]-self.xs[0]**2),
#             1-self.xs[0],
#             sqrt(90)*(self.xs[3]-self.xs[2]**2),
#             1-self.xs[2],
#             sqrt(10)*(self.xs[1]+self.xs[3]-2),
#             sqrt(10)*(self.xs[1]-self.xs[3])
#         ]
#         return self.get_expr_from_rs(rs,self.m)


