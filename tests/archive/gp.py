import numpy as np
from pydacefit.regr import LinearMean
from pymoo.algorithms.so_genetic_algorithm import SingleObjectiveGeneticAlgorithm
from pymop.problem import Problem

from pydacefit.kernel import Kernel, SquaredExponential


class GaussianProcess:

    def __init__(self, kernel, regr) -> None:
        super().__init__()

        self.kernel = kernel
        self.regr = regr

        self.n_rows = None
        self.n_vars = None
        self.n_targets = None

        self.normX = None
        self.Y = None
        self.C = None

        self.K = None
        self.C = None
        self.K_inv = None
        self.alpha = None
        self.error = None

    def fit(self, X, Y):

        if len(Y.shape) == 1:
            Y = Y[:, None]

        self.n_rows = X.shape[0]
        self.n_vars = X.shape[1]
        self.n_targets = Y.shape[1]

        self.normX = (X - self.meanX)
        self.Y = Y

        if self.kernel.theta is None:
            self.optimize()
        else:
            self._fit()

    def _fit(self):

        # calculate the kernel matrix and cholesky decomposition
        self.K = self.kernel.calc(self.X, self.X)
        self.K += np.eye(self.n_rows, self.n_rows) * (10 + self.n_rows) * 1e-10
        self.C = np.linalg.cholesky(self.K)

        # calculate the inverse using the C matrix
        self.K_inv = np.eye(self.n_rows, self.n_rows)
        self.K_inv = np.linalg.solve(self.C, self.K_inv)
        self.K_inv = np.linalg.solve(self.C.transpose(), self.K_inv)

        # fitting the regression parameters
        self.regr.fit(self.X, self.Y, self.K_inv)

        # get the error that needs to be fitted by the gp
        self.error = self.Y - self.regr.predict(self.X)
        self.alpha = self.K_inv @ self.Y

    def predict(self, X_hat):

        # predict the mean
        X_hat = (X_hat - self.meanX) / self.stdX
        K_star = self.kernel.calc(X_hat, self.X)
        Y_hat = np.matmul(K_star, self.alpha) + self.regr.predict(X_hat)
        Y_hat = self.meanY + (Y_hat * self.stdY)

        # predict the mse
        K_star_star = self.kernel.calc(X_hat, X_hat)
        gp_cov = K_star_star - np.matmul(np.matmul(K_star, self.K_inv), K_star.transpose())
        R = self.regr.get(X_hat).transpose() - np.matmul(np.matmul(self.regr.H, self.K_inv), K_star.transpose())
        regr_cov = R.transpose() @ self.regr.K_inv @ R
        mse = np.square(self.stdY) * np.mean(np.square(self.error)) * np.diagonal(gp_cov + regr_cov)

        return Y_hat, mse

    def get_neg_log_likelihood(self):
        complexity = np.log(np.square(np.prod(np.diagonal(self.C))))
        training_error = self.error.transpose() @ self.K_inv @ self.error
        neg_log_likelihood = - 0.5 * (complexity + training_error + self.n_rows * np.log(2 * np.pi))
        return neg_log_likelihood

    def get_neg_log_likelihood_gradient(self):

        pass

    def optimize(self):

        n_params = 1

        def evaluate(x, f):
            for i in range(x.shape[0]):
                self.kernel.theta = np.exp(x[i, :] * np.ones(10))
                self._fit()
                f[i, 0] = self.get_neg_log_likelihood()

        class HyperparameterProblem(Problem):
            def __init__(self):
                Problem.__init__(self)
                self.n_var = n_params
                self.n_constr = 0
                self.n_obj = 1
                self.func = self.evaluate_
                self.xl = 0.01 * np.ones(n_params)
                self.xu = 20 * np.ones(n_params)

            def evaluate_(self, x, f):
                evaluate(x, f)

        a = SingleObjectiveGeneticAlgorithm("real", pop_size=50, verbose=False)
        p = HyperparameterProblem()
        [X, _, _] = a.solve(p, 50000)

        self.kernel.theta = np.exp(X[0, :] * np.ones(10))
        self._fit()


if __name__ == "__main__":
    instance = "Rastrigin"
    X = np.loadtxt("/Users/julesy/workspace/dacefit-cpp/benchmark/instances/%s.x_train" % instance)
    X_hat = np.loadtxt("/Users/julesy/workspace/dacefit-cpp/benchmark/instances/%s.x_test" % instance)
    Y = np.loadtxt("/Users/julesy/workspace/dacefit-cpp/benchmark/instances/%s.f_train" % instance)
    # Y_hat = np.loadtxt("/Users/julesy/workspace/dacefit-cpp/benchmark/instances/%s.f_test"% instance)

    kernel = Kernel(SquaredExponential, np.array([0.5] * 10))
    regr = LinearMean()
    gp = GaussianProcess(kernel, regr)
    gp.fit(X, Y)
    gp.predict(X_hat)

    print(gp.get_neg_log_likelihood())
    # gp.optimize()
    # print(gp.get_neg_log_likelihood())
    print(gp.kernel.theta)

    GP_Y_hat, _ = gp.predict(X_hat)
    print(GP_Y_hat)

    # print(np.mean(np.abs(Y_hat - GP_Y_hat)))
