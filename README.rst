pydacefit
==================================

The documentation can be found here:
https://www.egr.msu.edu/coinlab/blankjul/pydacefit/

The purpose of this clone is to have a python version of the popular dacefit toolbox in MATLAB .
This framework is an exact clone of the original code and the correctness has been checked.

Installation
==================================

The test problems are uploaded to the PyPi Repository.

.. code:: bash

    pip install pydacefit

Usage
==================================

In general, the function calls are very similar to the MATLAB Version. The only difference is that in
Python an actual object is used which provide the functions fit and predict.

The following shows how to use this framework:

.. literalinclude:: ../../pydacefit/usage.py
   :language: python



Contact
==================================
Feel free to contact me if you have any question:

| Julian Blank (blankjul [at] egr.msu.edu)
| Michigan State University
| Computational Optimization and Innovation Laboratory (COIN)
| East Lansing, MI 48824, USA
