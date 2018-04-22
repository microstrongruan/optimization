import numpy as np
from scipy.optimize import line_search
DIFF = 2e-8


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# cholesky decomposition should be quicker
def is_pos_def_(x):
    try:
        np.linalg.cholesky(x)
        return True
    finally:
        return False

class Basic_Newton:
    def __init__(self, f, g, G):
        self.f = f
        self.g = g
        self.G = G

    @staticmethod
    def get_name():
        return "Basic_Newton"

    def optimize(self, start_point, verbose=False):
        xk = start_point
        iter = 0
        while True:
            dk = np.linalg.solve(self.G(xk), -self.g(xk))
            alpha = 1
            xk_plus_1 = xk + alpha * dk

            if verbose:
                print('----------')
                print("alpha", alpha)
                print("dk" , dk)
                print("x_k+1", xk_plus_1)
                print("f_k+1", self.f(xk_plus_1))
            if np.min(np.abs(xk - xk_plus_1)) < DIFF:
                break
            iter+=1
            xk = xk_plus_1

        print("     final point", xk_plus_1)
        print("     final_fval", self.f(xk_plus_1))
        print("     iter times", iter)
        return xk_plus_1, self.f(xk_plus_1)


class Zuni_Newton:
    def __init__(self, f, g, G):
        self.f = f
        self.g = g
        self.G = G

    @staticmethod
    def get_name():
        return "Zuni_Newton"

    def optimize(self, start_point, verbose=False):
        xk = start_point
        iter = 0
        while True:
            dk = np.linalg.solve(self.G(xk), -self.g(xk))
            alpha,fc,gc,new_fval,old_fval,new_slope = line_search(self.f, self.g, xk, dk)

            xk_plus_1 = xk + alpha*dk

            if verbose:
                print('----------')
                print("alpha", alpha)
                print("dk" , dk)
                print("x_k+1", xk_plus_1)
                print("f_k+1", self.f(xk_plus_1))

            if np.min(np.abs(xk-xk_plus_1)) < DIFF:
                break
            iter +=1
            xk = xk_plus_1

        print("     final point", xk_plus_1)
        print("     final_fval", self.f(xk_plus_1))
        print("     iter times", iter)
        return xk_plus_1, self.f(xk_plus_1)

class LM_Newton:
    def __init__(self, f, g, G, init_vk = 1e-5):
        self.f = f
        self.g = g
        self.G = G
        self.init_vk = init_vk

    @staticmethod
    def get_name():
        return "LM_Newton"


    def optimize(self, start_point, verbose=False):
        xk = start_point
        iter = 0
        while True:
            LM_G = self.G(xk)
            vk = np.eye(LM_G.shape[0], LM_G.shape[1])*self.init_vk
            while not is_pos_def(LM_G):
                LM_G += vk
                vk += vk
            dk = np.linalg.solve(LM_G, -self.g(xk))

            alpha,fc,gc,new_fval,old_fval,new_slope = line_search(self.f, self.g, xk, dk)

            xk_plus_1 = xk + alpha*dk

            if verbose:
                print('----------')
                print("alpha", alpha)
                print("dk" , dk)
                print("x_k+1", xk_plus_1)
                print("f_k+1", self.f(xk_plus_1))

            if np.min(np.abs(xk-xk_plus_1)) < DIFF:
                break
            iter += 1
            xk = xk_plus_1

        print("     final point", xk_plus_1)
        print("     final_fval", self.f(xk_plus_1))
        print("     iter times", iter)
        return xk_plus_1, self.f(xk_plus_1)

class SR1:
    def __init__(self, f, g, G):
        self.f = f
        self.g = g
        # G is not needed
        # self.G = G

    @staticmethod
    def get_name():
        return "SR1"


    def optimize(self, start_point, verbose=False):
        xk = start_point
        gk = self.g(xk)
        Hk = np.eye(gk.shape[0], gk.shape[0])
        iter =0
        while True:
            dk = - Hk.dot(gk)
            alpha,fc,gc,new_fval,old_fval,new_slope = line_search(self.f, self.g, xk, dk)

            #TODO
            if alpha == None:
                alpha = 0.01
            sk = alpha*dk
            xk_plus_1 = xk + sk
            gk_plus_1 = self.g(xk_plus_1)
            yk = gk_plus_1-gk

            # change to matrix, as column vector
            sk = np.array([sk]).T
            yk = np.array([yk]).T

            temp = sk-Hk.dot(yk)
            Hk_puls_1 = Hk + temp.dot(temp.T)/temp.T.dot(yk)

            if verbose:
                print('----------')
                print("alpha", alpha)
                print("dk" , dk)
                print("x_k+1", xk_plus_1)
                print("f_k+1", self.f(xk_plus_1))
                # print("Hk", Hk_puls_1)

            if np.min(np.abs(xk-xk_plus_1)) < DIFF:
                break
            iter+=1
            xk = xk_plus_1
            Hk = Hk_puls_1
            gk = gk_plus_1

        print("     final point", xk_plus_1)
        print("     final_fval", self.f(xk_plus_1))
        print("     iter times", iter)
        return xk_plus_1, self.f(xk_plus_1)

class DFP:
    def __init__(self, f, g, G):
        self.f = f
        self.g = g
        # G is not needed
        # self.G = G

    @staticmethod
    def get_name():
        return "DFP"


    def optimize(self, start_point, verbose=False):
        xk = start_point
        gk = self.g(xk)
        Hk = np.eye(gk.shape[0], gk.shape[0])
        iter = 0
        while True:
            dk = - Hk.dot(gk)
            alpha,fc,gc,new_fval,old_fval,new_slope = line_search(self.f, self.g, xk, dk)
            sk = alpha*dk
            xk_plus_1 = xk + sk
            gk_plus_1 = self.g(xk_plus_1)
            yk = gk_plus_1-gk

            # change to matrix, as column vector
            sk = np.array([sk]).T
            yk = np.array([yk]).T

            Hk_puls_1 = Hk +sk.dot(sk.T)/sk.T.dot(yk) - Hk.dot(yk).dot(yk.T).dot(Hk)/yk.T.dot(Hk).dot(yk)

            if verbose:
                print('----------')
                print("alpha", alpha)
                print("dk" , dk)
                print("x_k+1", xk_plus_1)
                print("f_k+1", self.f(xk_plus_1))
                # print("Hk", Hk_puls_1)

            if np.min(np.abs(xk-xk_plus_1)) < DIFF:
                break
            iter+=1
            xk = xk_plus_1
            Hk = Hk_puls_1
            gk = gk_plus_1

        print("     final point", xk_plus_1)
        print("     final_fval", self.f(xk_plus_1))
        print("     iter times", iter)
        return xk_plus_1, self.f(xk_plus_1)

class BFGS:
    def __init__(self, f, g, G):
        self.f = f
        self.g = g
        # G is not needed
        # self.G = G

    @staticmethod
    def get_name():
        return "BFGS"


    def optimize(self, start_point, verbose=False):
        xk = start_point
        gk = self.g(xk)
        Hk = np.eye(gk.shape[0], gk.shape[0])
        iter=0
        while True:
            dk = - Hk.dot(gk)
            alpha,fc,gc,new_fval,old_fval,new_slope = line_search(self.f, self.g, xk, dk)
            sk = alpha*dk
            xk_plus_1 = xk + sk
            gk_plus_1 = self.g(xk_plus_1)
            yk = gk_plus_1-gk

            # change to matrix, as column vector
            sk = np.array([sk]).T
            yk = np.array([yk]).T

            Hk_puls_1 = Hk + (1+yk.T.dot(Hk).dot(yk)/yk.T.dot(sk))*(sk.dot(sk.T)/yk.T.dot(sk))-\
                        (sk.dot(yk.T).dot(Hk)+Hk.dot(yk).dot(sk.T))/yk.T.dot(sk)

            if verbose:
                print('----------')
                print("alpha", alpha)
                print("dk" , dk)
                print("x_k+1", xk_plus_1)
                print("f_k+1", self.f(xk_plus_1))
                # print("Hk", Hk_puls_1)

            if np.min(np.abs(xk-xk_plus_1)) < DIFF:
                break
            iter+=1
            xk = xk_plus_1
            Hk = Hk_puls_1
            gk = gk_plus_1

        print("     final point", xk_plus_1)
        print("     final_fval", self.f(xk_plus_1))
        print("     iter_times", iter)
        return xk_plus_1, self.f(xk_plus_1)