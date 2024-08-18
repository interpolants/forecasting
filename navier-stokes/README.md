
Some simple code for Navier Stokes and CIFAR, implementing the method in 

[Probabilistic Forecasting with Stochastic Interpolants and Föllmer Processes](https://arxiv.org/abs/2403.13724). Yifan Chen, Mark Goldstein, Mengjian Hua, Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden.

This code is a quick demo version of code written collaboratively by the authors of the above work. This version emphasizes pedagogy + quick debugging + overfitting tests on a single GPU. A public release of the code from the above paper, including for video experiments, is forthcoming.

As for method and the code here:

- this method models data $X_1|X_0=x_0$ as the time 1 solution of an SDE whose initial condition is a point-mass at $x_0$. 
- This is accomplished by learning an SDE that models the interpolant $X_t = \alpha(t) X_0 + \beta(t) X_1 + \sigma(t) W_t$
-  $(X_0, X_1) \sim \rho(X_0)\rho(X_1|X_0)$ and $W_t$ is the Wiener process. 
- Here, $\alpha(0)=\beta(1)=1$ and $\alpha(1)=\beta(0)=\sigma(1)=0$. 
- Most often, $\alpha(t)=\sigma(t)=1-t$ and $\beta(t)=t$ or $\beta(t)=t^2$.
- The code learns the SDE drift $b$ 
- and then samples $dX_t = b_t(X, x_0) dt + \sigma(t) dW_t, \quad X_{t=0} = x_0$ in order to sample $X_1| X_0=x_0$.


The files in this repo are:
- interpolant.py: implements interpolants
- main.py: this runs an experiment and contains the trainer
- nse_data_tiny.pt: a very tiny subset of the Navier-stokes data from the paper
- README.md: you are here.  
- unet.py: the architecture. Thanks lucidrains!  
- utils.py: lots of random utils to make main.py work.

You can run the code with commands like:
```
python main.py --dataset nse --beta_fn t^2 --sigma_coef 1.0 --use_wandb 1 --debug 0 --overfit 0
```


Some notes:
- dataset can be cifar or nse. 
	- cifar is mostly meant for quick debugging, and we go from a constant initial condition since it is an unconditional task. Due to point mass, sigma_coef can be larger like 10.0
	- nse is navier-stokes. 
		- We go from $X_t$ to $X_{t+\tau}$.
		- The initial condition can optionally be lower resolution (i.e. for simultaneous forecasting and super-resolution). See lo_size and hi_size arguments in Config()
		- we include a tiny dataset here directly in the repo to keep the code runnable directly from git clone. The tiny dataset has 2 trajectories of 100 frames each which gets turned into many datapoints. More data is [here](https://zenodo.org/records/10939479). Each file is 200 trajectories. In some heavier code we have, we make separate dataloaders for each file and merge the iterators.

- sigma_coef is the factor $\epsilon$ in, for example $\sigma(t) = \epsilon*(1-t)$.

- $\beta(t) = t^2$ is the suggested coefficient to use (rather than, e.g. $\beta(t) = t$) because it means that $\dot \beta_0 = 0$ as discussed at the end of Section 3.2 and in appendix A.3). Empirically, this controls the variance of the loss (and the norm of the gradients) better.

- The code is currently parameterized in terms of the formulation in [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions
](https://arxiv.org/abs/2303.08797), where the interpolant is $x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) z$ where $z$ is Gaussian; to get the method from this work, $\gamma$ is set in a special way $\gamma(t)=\sigma(t)\sqrt{t}$ that is tied to the model diffusion coefficient $\sigma$ in the generative SDE $dX = bdt + \sigma(t) dW_t$. However, this code could be simplified to the presentation just in terms of $\sigma$, as above, with no mention of $\gamma$ in this special case.

- Set your wandb entity in the Config() in main py. 

- On wandb, you should see images of the form (initial condition, model sample, target). Vorticities for NSE and samples for CIFAR

- To speed up debugging, try --overfit 1, which trains on one batch. For cifar, you should see things that look like cifar samples after about 1000 gradient steps. 

```
python main.py --dataset cifar --beta_fn t^2 --sigma_coef 10.0 --use_wandb 1 --debug 0 --overfit 1
```
To be added soon are:
- the entropy/enstrophy spectra metrics and the error norm metric


If the method or the code is useful to you, kindly cite the following in any publications:
```
@article{chen2024probabilistic,
  title={Probabilistic Forecasting with Stochastic Interpolants and F$\backslash$" ollmer Processes},
  author={Chen, Yifan and Goldstein, Mark and Hua, Mengjian and Albergo, Michael S and Boffi, Nicholas M and Vanden-Eijnden, Eric},
  journal={arXiv preprint arXiv:2403.13724},
  year={2024}
}
```

