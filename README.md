# Blog Plots
This repo contains all the short scripts I have used to generate content for my blogs.

# Installation
1. Clone the repo to your local machine.

2. Create a virtual environment for the repo in your desired location:

```bash
python3 -m venv path/where/you/want/venv
```

3. Activate the virtual environment:

```bash
source path/where/you/want/venv/bin/activate
```
4. Navigate to the root directory of the repo and install the requirements.
```bash
pip install -r requirements.txt
```
At this point you should be able to run any script you want from the root directory. All
plots are generated through scripts. For example, to generate a plot of marginalized multivariate
distributions you simply run the following from the root of the repo:
```bash
python -m multi_variate_gaussians.marginalized_multivariate ./multi_variate_gaussians/configs/conditional_gaussian.yaml ./marginalized_new.png
```

This should generate the plot and save it in the root of the directory. Adjusting the last arg in the
call will adjust where the plot is saved.

Note:

1. python 3.8.10 was used to generate this code base.
2. In general I would advise against using `python -m venv` and rather advocate for [pyenv](https://github.com/pyenv/pyenv)

# Plots

Below is a list of the blog posts I have made. If you wish to generate any of the
plots present in a specific blog post click the hyperlinked name of the blog below to find
the necessary commands.

1. [Bayesian Optimization with Gaussian Processes Part 1 - Multivariate Gaussians](./docs/Multivariate_Gaussians.md)
2. [Bayesian Optimisation with Gaussian Processes Part 2 - Gaussian Process Regression](./docs/Gaussian_Process_Regression.md)
3. [Bayesian Optimization with Gaussian Processes Part 3 - Bayesian Optimization](./docs/Bayesian_Optimization.md)

# Issues:

The current code base isn't the best code. I was prioritizing getting something out. I will be
documenting the issues in the coming weeks and working through them. The main issue to be aware of
is that sometimes there can be numerical instability with a particular matrix inversion. The warning
message will be as follows:
```bash
RuntimeWarning: invalid value encountered in sqrt
  sigma = np.sqrt(np.diag(cov_cond))
```
If this pops up just run the command again. As all data points effecting the matrix inversion are
randomly generated. So you can just re-run until there is no warning.

Some other issues are:
* DRY
* docstrings
* Better module organisation
* Trim down requirements

# Contributing:

If your sanity lapses and you want to make contributions to the repo please follow this workflow:

1. Make an issue detailing your contribution
2. Create a branch on which you will make the contribution. *Give the branch the same name as the title of the issue.*
3. Before committing any code ensure that the pre-commit hooks have run. (Black, flake8)
4. PR your contribution and wait for a review.
5. Once approved it is up to you to squash merge the changes into master.
