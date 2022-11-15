# Constraint Assistance Games

This repository can be used to solve _Constrant Assistance Games_, a constrained version
of [Assistance Games / Cooperative Inverse Reinforcement Learning](https://arxiv.org/abs/1606.03137).

## Installation on Mac M1

Because of CPLEX, this repository currently requires python 3.7, on Mac M1 this can be installed via:

```bash
conda create -n <env-name>
conda activate <env-name>
conda config --env --set subdir osx-64
conda install python=3.7
```

Choose (A) or (B) depending on whether you have the professional version of CPLEX.

* **(A) If you have access to the professional version of CPLEX**, install it via the GUI. Then, install the python
  API: `pip install /Applications/CPLEX_Studio<xyz>/cplex/python/3.7/x86-64_osx`


* **(B) If you want to use the free (limited) version of CPLEX**, it can be installed
  via `conda install -c ibmdecisionoptimization cplex` (although this limits the problem size solvable).

Then install the remaining packages.

```
conda install numpy tqdm matplotlib tabulate colorama imagio
pip install enquiries pptree
conda install -c conda-forge gym
```

You can verify your install with:

```
cd <parent>/cags
python -m unittest discover -s ./tests/auto_tests
```