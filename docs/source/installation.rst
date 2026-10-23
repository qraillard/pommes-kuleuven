.. _installation:

Installation
============

Pre-requisites
--------------

* `Anaconda 3.8 distribution <https://www.anaconda.com/distribution/>`_ or `Miniconda3 distribution <https://docs.conda.io/en/latest/miniconda.html>`_
* To clone pommes' Gitlab repository, `Git <https://git-scm.com/downloads>`_ (On Windows, `Git for Windows <https://git-for-windows.github.io/>`_ is preferred)
* If you wish to install from source, the cpp compiler corresponding to pommes' python version is needed. On linux,
  ``sudo apt-get install -y gcc make g++``. On windows, follow the instructions at `WindowsCompiler https://wiki.python.org/moin/WindowsCompilers`
* A working docker installation. Detailed installation instructions for each platform available at `Docker install https://docs.docker.com/install/`
* A Gitlab account with at least guest level to install from source or at least reporter level to pull docker image

Installing from source
----------------------

Obtaining the source code
~~~~~~~~~~~~~~~~~~~~~~~~~

You can either :
    - directly download the source in a compressed folder from pommes' main gitlab page : `https://git.persee.mines-paristech.fr/energy-alternatives/pommes`
    - clone pommes' repository using git either with SSH
      ``git clone git@git.persee.mines-paristech.fr:energy-alternatives/pommes.git`` or with HTTPS
      ``git clone https://git.persee.mines-paristech.fr/energy-alternatives/pommes.git`` after having set up the corresponding
      authentication.


Creating the conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Open a command line tool in the pommes root folder, enter the ``ci`` folder, and then run the following :

.. code-block:: shell

    conda env create -f conda_env.yml


For more information on conda environments, please visit https://conda.io/docs/using/envs.html .

.. warning::
    On Windows, if the environment installation fails for one the pip installed package, please make sure that the cpp
    compiler was properly installed and that the folder containing the executables is on the PATH.

.. note::
    If a conda environment with the name pommes already exists, this will fail. In this case, it is possible to
    either remove the existing environment with ``conda remove --name pommes --all`` or to change the name of
    the environment to be created by editing it directly in the .yml file.

Installing pommes
~~~~~~~~~~~~~~~~~~~~~~~~

Activate the environment we have just created :

.. code-block:: shell

    conda activate pommes

Then, from the root folder of pommes, run :

.. code-block:: shell

    pip install .

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

From the root folder of pommes, run :

.. code-block:: shell

    pytest




Compiling a local version of the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Additional dependencies are needed to compile the documentation :
    - pandoc and graphviz. Can be installed on linux with ``sudo apt-get install pandoc graphviz graphviz-dev``. On
      windows, see installation instructions on their respective websites.
    - python packages listed in doc/requirements.txt

When the dependencies are installed, cd into the ``doc`` folder, and then run the following :

.. code-block:: shell

    make html

The resulting documentation can be accessed by opening the file ``doc\_build\html\index.html``
