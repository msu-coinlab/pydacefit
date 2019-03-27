pydacefit
==================================

The documentation can be found here:
https://www.egr.msu.edu/coinlab/blankjul/pydacefit/

The purpose of this clone is to have a python version of the popular dacefit toolbox in MATLAB .
The toolbox can be found `here <https://www.mathworks.com/matlabcentral/fileexchange/38530-surrogate-model-optimization-toolbox>`_.

This framework is an exact clone of the original code and the correctness has been checked.
Please contact me if you should be scenarios where the values are significantly different (10^6).

Installation
==================================

The test problems are uploaded to the PyPi Repository.

.. code:: bash

    pip install pydacefit

Usage
==================================

.. code:: python

    
    import numpy as np

    from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_expg, corr_spline, corr_spherical
    from pydacefit.dace import DACE, regr_linear, regr_quadratic
    from pydacefit.regr import regr_constant

    import matplotlib.pyplot as plt

    # -----------------------------------------------
    # Different ways of initialization
    # -----------------------------------------------

    # regression can be: regr_constant, regr_linear or regr_quadratic
    regression = regr_constant
    # regression = regr_linear
    # regression = regr_quadratic


    # then define the correlation (all possible correlations are shown below)
    # please have a look at the MATLAB document for more details
    correlation = corr_gauss
    # correlation = corr_cubic
    # correlation = corr_exp
    # correlation = corr_expg
    # correlation = corr_spline
    # correlation = corr_spherical
    # correlation = corr_cubic


    # This initializes a DACEFIT objective using the provided regression and correlation
    # because an initial theta is provided and also thetaL and thetaU the hyper parameter
    # optimization is done
    dacefit = DACE(regr=regression, corr=correlation,
                   theta=1.0, thetaL=0.00001, thetaU=100)

    # if no lower and upper bounds are defined, then no hyperparameter optimization is executed
    dacefit_no_hyperparameter_optimization = DACE(regr=regression, corr=correlation,
                                                  theta=1.0, thetaL=None, thetaU=None)

    # to turn on the automatic relevance detection use a vector for theta and define bounds
    dacefit_with_ard = DACE(regr=regression, corr=correlation,
                            theta=[1.0, 1.0], thetaL=[0.001, 0.0001], thetaU=[20, 20])


    # -----------------------------------------------
    # Create some data for the purpose of testing
    # -----------------------------------------------

    def fun(X):
        return np.sum(np.sin(X * 2 * np.pi), axis=1)


    X = np.random.random((20, 1))
    F = fun(X)

    # -----------------------------------------------
    # Fit the model with the data and predict
    # -----------------------------------------------

    # create the model and fit it
    dacefit.fit(X, F)

    # predict values for plotting
    _X = np.linspace(0, 1, 100)[:, None]
    _F = dacefit.predict(_X)

    # -----------------------------------------------
    # Plot the results
    # -----------------------------------------------

    plt.scatter(X, F, label="prediction")
    plt.plot(_X, _F, label="data")
    plt.legend()
    plt.show()

    print("MSE: ", np.mean(np.abs(fun(_X)[:, None] - _F)))

Contact
==================================
Feel free to contact me if you have any question:

| Julian Blank (blankjul [at] egr.msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA
