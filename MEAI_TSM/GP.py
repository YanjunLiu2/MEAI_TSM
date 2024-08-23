from __future__ import annotations
import math
import gpytorch


import torch
import random
from gpytorch.kernels import RBFKernel
import numpy as np
import copy

from gpytorch.distributions import MultivariateNormal as Normal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from scipy.spatial.distance import squareform
from sklearn.metrics import r2_score





import time
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

from botorch.exceptions.warnings import OptimizationWarning
from botorch.optim.numpy_converter import (
    TorchAttr,
    module_to_array,
    set_params_with_array,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from botorch.optim.utils import (
    _filter_kwargs,
    _get_extra_mll_args,
    _scipy_objective_and_grad,
)
from gpytorch import settings as gpt_settings
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from scipy.optimize import Bounds, minimize
from torch import Tensor
from torch.nn import Module
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.optimizer import Optimizer


ParameterBounds = Dict[str, Tuple[Optional[float], Optional[float]]]
TScipyObjective = Callable[
    [np.ndarray, MarginalLogLikelihood, Dict[str, TorchAttr]], Tuple[float, np.ndarray]
]
TModToArray = Callable[
    [Module, Optional[ParameterBounds], Optional[Set[str]]],
    Tuple[np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]],
]
TArrayToMod = Callable[[Module, np.ndarray, Dict[str, TorchAttr]], Module]


class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float



def fit_gpytorch_torch(
    ard_len,
    mll: MarginalLogLikelihood,
    gamma=0.1,
    projection=None,
    projection_len=None,
    l1=False,
    bounds: Optional[ParameterBounds] = None,
    optimizer_cls: Optimizer = Adam,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = True,
    approx_mll: bool = True,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    r"""Fit a gpytorch model by maximizing MLL with a torch optimizer.

    The model and likelihood in mll must already be in train mode.
    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        optimizer_cls: Torch optimizer to use. Must not require a closure.
        options: options for model fitting. Relevant options will be passed to
            the `optimizer_cls`. Additionally, options can include: "disp"
            to specify whether to display model fitting diagnostics and "maxiter"
            to specify the maximum number of iterations.
        track_iterations: Track the function values and wall time for each
            iteration.
        approx_mll: If True, use gpytorch's approximate MLL computation (
            according to the gpytorch defaults based on the training at size).
            Unlike for the deterministic algorithms used in fit_gpytorch_scipy,
            this is not an issue for stochastic optimizers.

    Returns:
        2-element tuple containing
        - mll with parameters optimized in-place.
        - Dictionary with the following key/values:
        "fopt": Best mll value.
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationIteration objects with information on each
        iteration. If track_iterations is False, will be empty.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_torch(mll)
        >>> mll.eval()
    """
    optim_options = {"maxiter": 100, "disp": True, "lr": 0.05}
    optim_options.update(options or {})
    exclude = optim_options.pop("exclude", None)
    if exclude is not None:
        mll_params = [
            t for p_name, t in mll.named_parameters() if p_name not in exclude
        ]
    else:
        mll_params = list(mll.parameters())
    optimizer = optimizer_cls(
        params=[{"params": mll_params}],
        **_filter_kwargs(optimizer_cls, **optim_options),
    )

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(mll, "named_parameters_and_constraints"):
        for param_name, _, constraint in mll.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    iterations = []
    t1 = time.time()

    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in mll.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **optim_options)
    )
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    while not stop:
        optimizer.zero_grad()
        with gpt_settings.fast_computations(log_prob=approx_mll):
            output = mll.model(*train_inputs)
            # we sum here to support batch mode
            args = [output, train_targets] + _get_extra_mll_args(mll)
            loss = -mll(*args).sum()
            if l1==True:
                ard = torch.diag(torch.squeeze(ard_len.reciprocal() ** 2))
                l1_loss=ard.abs().sum()*gamma
                loss+=l1_loss
            loss.backward(retain_graph=True)
        loss_trajectory.append(loss.item())
        for name, param in mll.named_parameters():
            param_trajectory[name].append(param.detach().clone())
        if optim_options["disp"] and (
            (i + 1) % 10 == 0 or i == (optim_options["maxiter"] - 1)
        ):
            print(f"Iter {i + 1}/{optim_options['maxiter']}: {loss.item()}")
        if track_iterations:
            iterations.append(OptimizationIteration(i, loss.item(), time.time() - t1))
        optimizer.step()
        # project onto bounds:
        if bounds_:
            for pname, param in mll.named_parameters():
                if pname in bounds_:
                    param.data = param.data.clamp(*bounds_[pname])
        i += 1
        stop = stopping_criterion.evaluate(fvals=loss.detach())
    info_dict = {
        "fopt": loss_trajectory[-1],
        "wall_time": time.time() - t1,
        "iterations": iterations,
    }
    del optimizer
    del loss
    return mll, info_dict


class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, initialization, rank=6, interval1=100.):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module_projection = RBFKernel(
            ard_num_dims=rank,
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1)
        )
        proj = torch.tensor(initialization, dtype=torch.float64)
        proj.detach_().requires_grad_()
        self.register_parameter(
            "projection", torch.nn.Parameter(proj)
        )
        self.covar_module_ard = RBFKernel(
            ard_num_dims=train_x.shape[-1],
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1)
        )
        self.covar_warp = gpytorch.kernels.ScaleKernel(self.covar_module_ard)



    def forward(self, x):
        proj_x = x.matmul(self.projection)

        mean_x = self.mean_module(x)

        # this kernel is exp(-l_1^2 (x - x')P P^T(x - x') - l_2^2 (x - x')D(x - x'))
        # because we compute the product elementwise
        covar_x = self.covar_module_projection(proj_x) * self.covar_warp(x)#self.covar_module_ard(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DirichletGPModelARD(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, rank=6, interval1=100.):
        super(DirichletGPModelARD, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module_ard = RBFKernel(
            ard_num_dims=train_x.shape[-1],
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1)
        )
        self.covar_warp = gpytorch.kernels.ScaleKernel(self.covar_module_ard)



    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_warp(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPmodel(ExactGP):
    def __init__(self, train_x, train_y, likelihood,initialization, rank=6, interval1=100.):
        super(ExactGPmodel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module_projection = RBFKernel(
            ard_num_dims=rank,
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1)
        )
        proj = torch.tensor(initialization, dtype=torch.float64)
        proj.detach_().requires_grad_()
        self.register_parameter(
            "projection", torch.nn.Parameter(proj)
        )
        self.covar_module_ard = RBFKernel(
            ard_num_dims=train_x.shape[-1],
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1)
        )
        self.covar_warp = gpytorch.kernels.ScaleKernel(self.covar_module_ard)

    def forward(self, x):
        proj_x = x.matmul(self.projection)

        mean_x = self.mean_module(x)

        # this kernel is exp(-l_1^2 (x - x')P P^T(x - x') - l_2^2 (x - x')D(x - x'))
        # because we compute the product elementwise
        covar_x = self.covar_module_projection(proj_x) * self.covar_warp(x)  # self.covar_module_ard(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def make_and_fit_regressor(train_x, train_y, inbuffer, maxiter=2000, lr=0.1, rank=30, interval1=100.):
    likelihood = GaussianLikelihood()
    model = ExactGPmodel(train_x, train_y, likelihood, initialization=inbuffer, rank=rank, interval1=interval1)
    model = model.to(train_x)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    _, info_dict = fit_gpytorch_torch(model.projection, model.covar_module_projection.lengthscale,
                                      model.covar_module_ard.lengthscale, mll, options={"maxiter": 1000, "lr": lr})
    print("Final MLL: ", info_dict["fopt"])

    return model, info_dict["fopt"]



def make_and_fit_classifier(train_x, train_y, inbuffer, maxiter=2000, lr=0.1, rank=6, interval1=100.):
    likelihood = DirichletClassificationLikelihood(train_y, alpha_epsilon=0.01, learn_additional_noise=True)
    model = DirichletGPModel(
        train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes, initialization=inbuffer,
        rank=rank, interval1=interval1
    )
    model = model.to(train_x)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _, info_dict = fit_gpytorch_torch(model.covar_module_ard.lengthscale, mll, options={"maxiter": 1000, "lr": lr})
    print("Final MLL: ", info_dict["fopt"])

    return model, info_dict["fopt"]




def make_and_fit_classifier_ard(train_x, train_y, maxiter=2000, lr=0.1, rank=6, interval1=100.):
    likelihood = DirichletClassificationLikelihood(train_y, alpha_epsilon=0.01, learn_additional_noise=True)
    model = DirichletGPModelARD(
        train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes,
        rank=rank, interval1=interval1
    )
    model = model.to(train_x)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _, info_dict = fit_gpytorch_torch(model.covar_module_ard.lengthscale, mll, l1=True, options={"maxiter": 1000, "lr": lr})
    print("Final MLL: ", info_dict["fopt"])

    return model, info_dict["fopt"]




def compute_accuracy(model, likelihood, test_x, test_y, train_x, train_y):
    likelihood.eval()
    model.eval()

    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(test_x)

        pred_means = test_dist.loc

        y_pred = []

        for i in range(len(pred_means[0])):
            if pred_means[0][i] > pred_means[1][i]:
                y_pred.append(0.)
            else:
                y_pred.append(1.)
    count = 0.0
    count0 = 0.0
    count1 = 0.0
    tot0=0.
    tot1=0.
    testy = test_y.float()
    for i in range(len(testy)):
        if testy[i]==1:
            tot1=tot1+1.
        elif testy[i]==0.:
            tot0=tot0+1.
        if y_pred[i] == testy[i]:
            count = count + 1.
            if y_pred[i] == 1.:
                count1=count1+1
            else:
                count0 = count0+1
    acc0 = count0 / tot0
    acc1 = count1 / tot1
    acct = (count1 + count0) / (tot1 + tot0)

    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist2 = model(train_x)

        pred_means = test_dist2.loc

        y_pred2 = []

        for i in range(len(pred_means[0])):
            if pred_means[0][i] > pred_means[1][i]:
                y_pred2.append(0.)
            else:
                y_pred2.append(1.)
    count2 = 0.0
    testy2 = train_y.float()
    for i in range(len(testy2)):
        if y_pred2[i] == testy2[i]:
            count2 = count2 + 1.
    accuracy2 = count2 / train_x.shape[0]
    return acct, accuracy2, acc0, acc1


def model_score(model, likelihood, test_x, test_y, train_x, train_y):
    likelihood.eval()
    model.eval()
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        pred_y=likelihood(model(test_x)).mean.cpu().numpy()
        test_y=test_y.cpu().numpy()
        print(len(pred_y))
        test_r2=r2_score(test_y, pred_y)
        pred_y=likelihood(model(train_x)).mean.cpu().numpy()
        train_y=train_y.cpu().numpy()
        train_r2=r2_score(train_y,pred_y)
    return test_r2, train_r2


def cluster_lengthscales(model, thresh_pct=0.15):
    # construct learned covariance matrix
    #print(model.projection.cpu().data)
    #print(model.covar_module_projection.lengthscale.cpu().data)
    #print(model.covar_module_ard.lengthscale.cpu().data)
    print('Scale', model.covar_warp.outputscale.cpu().data)

    cvector = model.projection.cpu().data.div(model.covar_module_projection.lengthscale.cpu().data)
    estimated_covar = model.projection / (model.covar_module_projection.lengthscale ** 2) @ model.projection.t() + \
                      torch.diag(torch.squeeze(model.covar_module_ard.lengthscale.reciprocal() ** 2))

    covar_inv_diags = estimated_covar.diag()  # ** 0.5
    estimated_corr = estimated_covar / torch.outer(covar_inv_diags ** 0.5, covar_inv_diags ** 0.5)

    estimated_dist = covar_inv_diags.unsqueeze(0) + covar_inv_diags.unsqueeze(1) - 2. * estimated_covar

    return estimated_covar, estimated_corr, model.projection.cpu().detach().numpy(), model.covar_module_projection.lengthscale.cpu().detach().numpy(), model.covar_module_ard.lengthscale.cpu().detach().numpy(), model.mean_module.constant.cpu().detach().numpy()


def ard_lengthscales(model):
    print('Scale', model.covar_warp.outputscale.cpu().data)
    ard_matrix=torch.diag(torch.squeeze(model.covar_module_ard.lengthscale.reciprocal() ** 2))
    return ard_matrix, model.mean_module.constant.cpu().detach().numpy()

def sign(number):
    if number>=0:
        return 1.
    else:
        return -1.

def rescale(matrix):

    for ri in range(len(matrix)):
        for rj in range(len(matrix[ri])):
            if abs(matrix[ri][rj])>0.8:
                matrix[ri][rj]=sign(matrix[ri][rj])*0.4+sign(matrix[ri][rj])*(abs(matrix[ri][rj])-0.8)*3.
            else:
                matrix[ri][rj]=matrix[ri][rj]/2.
    return matrix


def ele_probability(rank, device, proj_list, prolen_list, ardlen_list, mean_list, allx, ally, ele_features):

    oldproj0 = np.mean(proj_list, axis=0)
    oldproj0 = torch.tensor(oldproj0, dtype=torch.float64).to(device)
    oldprolen0 = np.mean(prolen_list, axis=0)
    oldprolen0 = torch.tensor(oldprolen0, dtype=torch.float64).to(device)
    oldardlen0 = np.mean(ardlen_list, axis=0)
    oldardlen0 = torch.tensor(oldardlen0, dtype=torch.float64).to(device)
    mean0 = np.mean(mean_list, axis=0)
    mean0 = torch.tensor(mean0, dtype=torch.float64).to(device)

    likelihood = likelihood = DirichletClassificationLikelihood(ally, learn_additional_noise=True)
    model = DirichletGPModel(
        allx, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes,
        initialization=oldproj0,
        rank=rank, interval1=100.
    )

    model.covar_module_projection.lengthscale = oldprolen0
    model.covar_module_ard.lengthscale = oldardlen0
    with torch.no_grad():
        model.mean_module.constant[0][0] = mean0[0][0]
        model.mean_module.constant[1][0] = mean0[1][0]
        model = model.to(device)
    model.double()
    likelihood.double()
    model.eval()
    likelihood.eval()

    ele_prob=[]
    for i in range(len(ele_features)):
        point=torch.tensor([ele_features[i]], dtype=torch.float64)
        point=point.to(device)
        with gpytorch.settings.fast_pred_var(), torch.no_grad():
            test_dist = model(point)
            pred_samples = test_dist.sample(torch.Size((256,))).exp()
            probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
            std_probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).std(0)
            ele_prob.append(probabilities[1].cpu().data.numpy())

    return ele_prob