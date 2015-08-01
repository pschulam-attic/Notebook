import numpy as np
from lbfgs import LBFGS, LBFGSError


class OWLQN:
    def __init__(self, f, g, penalty, x0, **kwargs):
        self.f = f
        self.g = g
        self.penalty = penalty
        self.x = x0.copy()

        self.optimizer = LBFGS()
        self.optimizer.orthantwise_c = self.penalty
        self.optimizer.linesearch = 'wolfe'
        self.optimizer.max_iterations = kwargs.get('max_iterations', 100)
        self.optimizer.delta = kwargs.get('delta', 1e-4)
        self.optimizer.past = 1

    def minimize(self):

        def objective(x, g):
            y = self.f(x)
            g[:] = self.g(x)
            return y

        def progress(x, g, f_x, xnorm, gnorm, step, k, ls):
            self.x[:] = x

        try:
            self.optimizer.minimize(objective, self.x, progress)
        except KeyError as e:
            pass
        except LBFGSError as e:
            pass

        return self
