.. _how_to_use:

How To Use
==========

This is a python package (i.e. a collection of subpackages and modules accompanied by an installer). As such, it can be
used in three main ways :

    - imported in a user-defined script and launched by the python interpreter :

    .. code-block:: shell

        python my_script.py

    For beginners, it may be easier to start from one of the tens of test scripts available from the /tests folder

    - imported in another package :

    .. code-block:: python

        from pommes.model import build_model

    - interactively in the Ipython Notebook launched using :

    .. code-block:: shell

        jupyter notebook

    several example notebooks are available alongside the documentation.
