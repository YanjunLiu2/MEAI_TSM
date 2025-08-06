
from __future__ import annotations
import math
import torch
import gpytorch
from matplotlib.ticker import FormatStrFormatter
import torch

from gpytorch.kernels import RBFKernel
import xlrd
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import DirichletClassificationLikelihood
from gpytorch.kernels import RBFKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import squareform
import xlwt
import scipy.cluster.hierarchy as spc





import time
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import numpy as np
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
    projection,
    projection_len,
    ard_len,
    mll: MarginalLogLikelihood,
    gamma=0.0001,
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

            """estimated_covar = projection / (
                projection_len ** 2) @ projection.t() + \
                torch.diag(torch.squeeze(ard_len.reciprocal() ** 2))

            covar_inv_diags = estimated_covar.diag()  # ** 0.5
            estimated_corr = estimated_covar / torch.outer(covar_inv_diags ** 0.5, covar_inv_diags ** 0.5)

            l1_loss=estimated_corr.abs().sum()*gamma"""

            loss = -mll(*args).sum()#+l1_loss
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
    return mll, info_dict



def generate_data(group, file_loc=r"16Mdata1.xls"):
    # X data generation
    excel = xlrd.open_workbook(file_loc)

    all_sheet = excel.sheets()

    # read all the data
    database = all_sheet[0]
    allresult = database.col_values(0)[3:]
    allmatname = database.col_values(1)[3:]
    alldsq = database.col_values(2)[3:]
    alldv = database.col_values(3)[3:]
    allsqatom = database.col_values(8)[3:]
    for i in range(len(allsqatom)):
        if allsqatom[i] == 'D':
            allsqatom[i] = 'H'
    alladata = database.col_values(10)[3:]
    allcdata = database.col_values(11)[3:]
    allscdata = database.col_values(12)[3:]
    structure = database.col_values(13)[3:]

    # generate subgroup data based on different structure types
    result = [[] for n in range(len(group))]
    matname = [[] for n in range(len(group))]
    dsq = [[] for n in range(len(group))]
    dv = [[] for n in range(len(group))]
    sqatom = [[] for n in range(len(group))]
    adata = [[] for n in range(len(group))]
    cdata = [[] for n in range(len(group))]

    for n in range(len(structure)):
        for k in range(len(group)):
            #tip=0
            for l in range(len(group[k])):
                if structure[n] == group[k][l]:
                    result[k].append(allresult[n])
                    matname[k].append(allmatname[n])
                    dsq[k].append(alldsq[n])
                    dv[k].append(alldv[n])
                    sqatom[k].append(allsqatom[n])
                    adata[k].append(alladata[n])
                    cdata[k].append(allcdata[n])
                    tip=1


    # read atomic database
    excel1 = xlrd.open_workbook(r"atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    excel2 = xlrd.open_workbook(r"xenonpy.xls")
    all_sheet2 = excel2.sheets()
    xenondata = all_sheet2[0]
    polar = xenondata.col_values(57)
    fcc = xenondata.col_values(25)
    excel3 = xlrd.open_workbook(r"Econfig.xls")
    all_sheet3 = excel3.sheets()
    edata = all_sheet3[0]
    ele2 = edata.col_values(0)[1:]
    valence = edata.col_values(1)[1:]

    Xtot = []
    Ytot = []
    composition = []
    elist = []
    for i in range(len(group)):
        Xtot.append([])
        Ytot.append([])
        composition.append([])
        elist.append([])

    for n in range(len(group)):
        labels = [x[0] == "y" for x in result[n]]
        for i in range(len(matname[n])):
            elements = []
            ratio = []
            for j in range(len(matname[n][i])):
                if (j < len(matname[n][i]) - 1) and (matname[n][i][j] == r')') and (
                        ord(matname[n][i][j + 1]) > 47) and (
                        ord(matname[n][i][j + 1]) < 58):
                    print(i, matname[n][i])
                if (ord(matname[n][i][j]) > 64) & (ord(matname[n][i][j]) < 91):
                    ele = matname[n][i][j]
                    if j == (len(matname[n][i]) - 1):
                        rate = 1.
                    if j < (len(matname[n][i]) - 1):
                        if (ord(matname[n][i][j + 1]) > 96) & (ord(matname[n][i][j + 1]) < 123):
                            ele = ele + matname[n][i][j + 1]
                            if (j + 2 == len(matname[n][i])):
                                rate = 1.
                            elif ((ord(matname[n][i][j + 2]) > 64) & (ord(matname[n][i][j + 2]) < 91)) or (
                                    matname[n][i][j + 2] == r' ') or (matname[n][i][j + 2] == r'(') or (
                                    matname[n][i][j + 2] == r')'):
                                rate = 1.
                            else:
                                step = 2
                                string = ''
                                while (j + step < len(matname[n][i])) and ((ord(matname[n][i][j + step]) == 46) or (
                                        (ord(matname[n][i][j + step]) > 47) and (ord(matname[n][i][j + step]) < 58))):
                                    string = string + matname[n][i][j + step]
                                    step = step + 1
                                rate = float(string)
                        elif ((ord(matname[n][i][j + 1]) > 64) & (ord(matname[n][i][j + 1]) < 91)) or (
                                matname[n][i][j + 1] == r' ') or (matname[n][i][j + 1] == r'(') or (
                                matname[n][i][j + 1] == r')'):
                            rate = 1.
                        else:
                            step = 1
                            string = ''
                            while (j + step < len(matname[n][i])) and ((ord(matname[n][i][j + step]) == 46) or (
                                    (ord(matname[n][i][j + step]) > 47) and (ord(matname[n][i][j + step]) < 58))):
                                string = string + matname[n][i][j + step]
                                step = step + 1
                            rate = float(string)
                    rep = 0
                    for k in range(len(elements)):
                        if (elements[k] == ele):
                            rep = 1
                            ratio[k] = ratio[k] + rate
                    if (rep == 0):
                        elements.append(ele)
                        ratio.append(rate)
            composition[n].append(ratio)
            elist[n].append(elements)


            ea, ip, en, rc, ve, pl = [], [], [], [], [], []
            tv = 0.
            for j in range(len(elements)):
                for k in range(len(ele2)):
                    if elements[j] == ele2[k]:
                        tv = tv + float(ratio[j]) * float(valence[k])

            for j in range(len(elements)):
                for k in range(len(ele2)):
                    if elements[j] == ele2[k]:
                        pl.append(polar[k])
                        ve.append(valence[k])

            for j in range(len(elements)):
                for k in range(len(ele1)):
                    if (elements[j] == ele1[k]):
                        ip.append(eleip1[k])
                        ea.append(elena1[k])
                        en.append(elene1[k])
                        rc.append(rcov1[k])

            for k in range(len(ele1)):
                if sqatom[n][i] == ele1[k]:
                    ipsq = eleip1[k]
                    easq = elena1[k]
                    ensq = elene1[k]
                    rcsq = rcov1[k]

            for k in range(len(ele2)):
                if sqatom[n][i] == ele2[k]:
                    plsq = polar[k]
                    vesq = valence[k]
                    fccsq = fcc[k]

            datapoint = [ensq, fccsq]
            #[max(rc), min(ea), easq, min(en), ensq, max(ve), min(ve), vesq, tv,
                        # dsq[n][i], dv[n][i], fccsq]
            #if (dsq[n][i] > 2.25) and (dsq[n][i] < 3.75) and (dv[n][i] < 3.5) and (cdata[n][i] < 15.) and (max(rc) < 220) and (
                    #min(rc) > 40):
            Xtot[n].append(datapoint)
            Ytot[n].append(labels[i])

        print(len(Xtot[n]), len(Xtot[n][0]))


    X1 = [[] for i in range(len(group))]
    X0 = [[] for i in range(len(group))]
    for n in range(len(group)):
        count1 = 0
        count0 = 0
        for i in range(len(Ytot[n])):
            if Ytot[n][i] == 1:
                X1[n].append(Xtot[n][i])
                count1 = count1 + 1
            else:
                X0[n].append(Xtot[n][i])
                count0 = count0 + 1
        print(n, count1, count0)



    return X1, X0



def fixed_data(group, file_loc=r"16Mdata1.xls"):
    # X data generation
    excel = xlrd.open_workbook(file_loc)

    all_sheet = excel.sheets()

    # read all the data
    database = all_sheet[0]
    allresult = database.col_values(0)[3:]
    allmatname = database.col_values(1)[3:]
    alldsq = database.col_values(2)[3:]
    alldv = database.col_values(3)[3:]
    allsqatom = database.col_values(8)[3:]
    for i in range(len(allsqatom)):
        if allsqatom[i] == 'D':
            allsqatom[i] = 'H'
    alladata = database.col_values(10)[3:]
    allcdata = database.col_values(11)[3:]
    allscdata = database.col_values(12)[3:]
    structure = database.col_values(13)[3:]

    # generate subgroup data based on different structure types
    result = [[] for n in range(len(group))]
    matname = [[] for n in range(len(group))]
    dsq = [[] for n in range(len(group))]
    dv = [[] for n in range(len(group))]
    sqatom = [[] for n in range(len(group))]
    adata = [[] for n in range(len(group))]
    cdata = [[] for n in range(len(group))]

    for n in range(len(structure)):
        for k in range(len(group)):
            #tip=0
            for l in range(len(group[k])):
                if structure[n] == group[k][l]:
                    result[k].append(allresult[n])
                    matname[k].append(allmatname[n])
                    dsq[k].append(alldsq[n])
                    dv[k].append(alldv[n])
                    sqatom[k].append(allsqatom[n])
                    adata[k].append(alladata[n])
                    cdata[k].append(allcdata[n])
                    tip=1


    # read atomic database
    excel1 = xlrd.open_workbook(r"atomic.xls")
    all_sheet1 = excel1.sheets()
    atomicdata = all_sheet1[0]
    ele1 = atomicdata.col_values(0)
    elene1 = atomicdata.col_values(1)
    elena1 = atomicdata.col_values(2)
    eleip1 = atomicdata.col_values(3)
    rcov1 = atomicdata.col_values(4)
    excel2 = xlrd.open_workbook(r"xenonpy.xls")
    all_sheet2 = excel2.sheets()
    xenondata = all_sheet2[0]
    polar = xenondata.col_values(57)
    fcc = xenondata.col_values(25)
    excel3 = xlrd.open_workbook(r"Econfig.xls")
    all_sheet3 = excel3.sheets()
    edata = all_sheet3[0]
    ele2 = edata.col_values(0)[1:]
    valence = edata.col_values(1)[1:]

    Xtot = []
    Ytot = []
    composition = []
    elist = []
    for i in range(len(group)):
        Xtot.append([])
        Ytot.append([])
        composition.append([])
        elist.append([])

    for n in range(len(group)):
        labels = [x[0] == "y" for x in result[n]]
        for i in range(len(matname[n])):
            elements = []
            ratio = []
            for j in range(len(matname[n][i])):
                if (j < len(matname[n][i]) - 1) and (matname[n][i][j] == r')') and (
                        ord(matname[n][i][j + 1]) > 47) and (
                        ord(matname[n][i][j + 1]) < 58):
                    print(i, matname[n][i])
                if (ord(matname[n][i][j]) > 64) & (ord(matname[n][i][j]) < 91):
                    ele = matname[n][i][j]
                    if j == (len(matname[n][i]) - 1):
                        rate = 1.
                    if j < (len(matname[n][i]) - 1):
                        if (ord(matname[n][i][j + 1]) > 96) & (ord(matname[n][i][j + 1]) < 123):
                            ele = ele + matname[n][i][j + 1]
                            if (j + 2 == len(matname[n][i])):
                                rate = 1.
                            elif ((ord(matname[n][i][j + 2]) > 64) & (ord(matname[n][i][j + 2]) < 91)) or (
                                    matname[n][i][j + 2] == r' ') or (matname[n][i][j + 2] == r'(') or (
                                    matname[n][i][j + 2] == r')'):
                                rate = 1.
                            else:
                                step = 2
                                string = ''
                                while (j + step < len(matname[n][i])) and ((ord(matname[n][i][j + step]) == 46) or (
                                        (ord(matname[n][i][j + step]) > 47) and (ord(matname[n][i][j + step]) < 58))):
                                    string = string + matname[n][i][j + step]
                                    step = step + 1
                                rate = float(string)
                        elif ((ord(matname[n][i][j + 1]) > 64) & (ord(matname[n][i][j + 1]) < 91)) or (
                                matname[n][i][j + 1] == r' ') or (matname[n][i][j + 1] == r'(') or (
                                matname[n][i][j + 1] == r')'):
                            rate = 1.
                        else:
                            step = 1
                            string = ''
                            while (j + step < len(matname[n][i])) and ((ord(matname[n][i][j + step]) == 46) or (
                                    (ord(matname[n][i][j + step]) > 47) and (ord(matname[n][i][j + step]) < 58))):
                                string = string + matname[n][i][j + step]
                                step = step + 1
                            rate = float(string)
                    rep = 0
                    for k in range(len(elements)):
                        if (elements[k] == ele):
                            rep = 1
                            ratio[k] = ratio[k] + rate
                    if (rep == 0):
                        elements.append(ele)
                        ratio.append(rate)
            composition[n].append(ratio)
            elist[n].append(elements)


            ea, ip, en, rc, ve, pl = [], [], [], [], [], []
            tv = 0.
            for j in range(len(elements)):
                for k in range(len(ele2)):
                    if elements[j] == ele2[k]:
                        tv = tv + float(ratio[j]) * float(valence[k])

            for j in range(len(elements)):
                for k in range(len(ele2)):
                    if elements[j] == ele2[k]:
                        pl.append(polar[k])
                        ve.append(valence[k])

            for j in range(len(elements)):
                for k in range(len(ele1)):
                    if (elements[j] == ele1[k]):
                        ip.append(eleip1[k])
                        ea.append(elena1[k])
                        en.append(elene1[k])
                        rc.append(rcov1[k])

            for k in range(len(ele1)):
                if sqatom[n][i] == ele1[k]:
                    ipsq = eleip1[k]
                    easq = elena1[k]
                    ensq = elene1[k]
                    rcsq = rcov1[k]

            for k in range(len(ele2)):
                if sqatom[n][i] == ele2[k]:
                    plsq = polar[k]
                    vesq = valence[k]
                    fccsq = fcc[k]

            datapoint = [ensq, fccsq]
            #[max(rc), min(ea), easq, min(en), ensq, max(ve), min(ve), vesq, tv,
                        # dsq[n][i], dv[n][i], fccsq]
            #if (dsq[n][i] > 2.25) and (dsq[n][i] < 3.75) and (dv[n][i] < 3.5) and (cdata[n][i] < 15.) and (max(rc) < 220) and (
                    #min(rc) > 40):
            Xtot[n].append(datapoint)
            Ytot[n].append(labels[i])

        print(len(Xtot[n]), len(Xtot[n][0]))

    X1 = [[] for i in range(len(group))]
    X0 = [[] for i in range(len(group))]
    for n in range(len(group)):
        count1 = 0
        count0 = 0
        for i in range(len(Ytot[n])):
            if Ytot[n][i] == 1:
                X1[n].append(Xtot[n][i])
                count1 = count1 + 1
            else:
                X0[n].append(Xtot[n][i])
                count0 = count0 + 1
        print(n, count1, count0)



    fixed_x=[]
    for i in range(2):
        fixed_x.append(0.)
    for n in range(2):
        for i in range(len(X0[n])):
            for j in range(2):
                fixed_x[j]=fixed_x[j]+X0[n][i][j]
        for i in range(len(X1[n])):
            for j in range(2):
                fixed_x[j]=fixed_x[j]+X1[n][i][j]

    for i in range(2):
        fixed_x[i]=fixed_x[i]/float(len(X0[0])+len(X0[1])+len(X1[0])+len(X1[1]))

    return fixed_x




class DirichletGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes, initialization, rank=4, interval1=5.):
        super(DirichletGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module_projection = RBFKernel(
            ard_num_dims=rank,
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1)
        )
        proj = torch.tensor(initialization, dtype=torch.float32)
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



def make_and_fit_classifier(train_x, train_y, inbuffer, maxiter=2000, lr=0.1, rank=2, interval1=100.):
    likelihood = DirichletClassificationLikelihood(train_y, alpha_epsilon=0.01, learn_additional_noise=True)
    model = DirichletGPModel(
        train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes, initialization=inbuffer,
        rank=rank, interval1=interval1
    )
    model = model.to(train_x)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _, info_dict = fit_gpytorch_torch(model.projection, model.covar_module_projection.lengthscale, model.covar_module_ard.lengthscale, mll, options={"maxiter": 1000, "lr": lr})
    print("Final MLL: ", info_dict["fopt"])

    return model, info_dict["fopt"]

def make_and_fit_classifier_s(train_x, train_y, inbuffer, projection_lengthscale, ard_lengthscale, mean0, maxiter=2000, lr=0.008, rank=2, interval1=15.):
    likelihood = DirichletClassificationLikelihood(train_y, learn_additional_noise=True)
    model = DirichletGPModel(
        train_x, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes, initialization=inbuffer,
        rank=rank, interval1=interval1
    )
    model.covar_module_projection.lengthscale = projection_lengthscale
    model.covar_module_ard.lengthscale = ard_lengthscale
    model.mean_module.constant = mean0
    model = model.to(train_x)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    _, info_dict = fit_gpytorch_torch(model.projection, model.covar_module_projection.lengthscale, model.covar_module_ard.lengthscale, mll, options={"maxiter": 1000, "lr": lr})
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


def cluster_lengthscales(model, thresh_pct=0.15):
    # construct learned covariance matrix
    print(model.projection.cpu().data)
    print(model.covar_module_projection.lengthscale.cpu().data)
    print(model.covar_module_ard.lengthscale.cpu().data)

    cvector = model.projection.cpu().data.div(model.covar_module_projection.lengthscale.cpu().data)
    estimated_covar = model.projection / (model.covar_module_projection.lengthscale ** 2) @ model.projection.t() + \
                      torch.diag(torch.squeeze(model.covar_module_ard.lengthscale.reciprocal() ** 2))

    covar_inv_diags = estimated_covar.diag()  # ** 0.5
    estimated_corr = estimated_covar / torch.outer(covar_inv_diags ** 0.5, covar_inv_diags ** 0.5)

    estimated_dist = covar_inv_diags.unsqueeze(0) + covar_inv_diags.unsqueeze(1) - 2. * estimated_covar

    return estimated_covar, estimated_corr, model.projection.cpu().detach().numpy(), model.covar_module_projection.lengthscale.cpu().detach().numpy(), model.covar_module_ard.lengthscale.cpu().detach().numpy(), model.mean_module.constant.cpu().detach().numpy()


device = torch.device('cuda:0')
detype = torch.double
print(torch.__version__)
basis = torch.randn(2, 2, dtype=torch.float64) / 2.
group=[['PbFCl','ZrSiS-UP2','ZrSiTe','AmTe2-x','PrOI','Cu2Sb'],['ZrCuSiAs-HfCuSi2','LaZn0.5Sb2'],['CaBe2Ge2']]
X1,X0 = generate_data(group)
fixdata=fixed_data(group)
print('fix',fixdata)
#second training set

buffer1 = torch.tensor(X1[0]+X1[1], dtype=torch.float64)
buffer0 = torch.tensor(X0[0]+X0[1], dtype=torch.float64)
buffer1 = buffer1.to(device)
buffer0 = buffer0.to(device)



acc_list, acc_list2, state_dict_list, mll_list, acc0_list, acc1_list, proj_list, prolen_list, ardlen_list, mean_list = [], [], [], [], [], [], [], [], [], []
summatrix = torch.zeros([2, 2], dtype=torch.float)
summatrix = summatrix.to(device)
rawmatrix = torch.zeros([2, 2], dtype=torch.float)
rawmatrix = rawmatrix.to(device)

r1 = torch.zeros([2, 2], dtype=torch.float)
r1 = rawmatrix.to(device)
for i in range(60):
    print("trying seed: ", i)
    torch.random.manual_seed(i+14)

    # note that now the training set is not fixed
    # lets use 80%
    shuffled_inds0 = torch.randperm(buffer0.shape[0])
    shuffled_inds1 = torch.randperm(buffer1.shape[0])
    cv1 = int(float(len(buffer1))*0.8)
    cv0 = int(float(len(buffer0))*0.8)
    trainset0 = shuffled_inds0[:cv0]
    testset0 = shuffled_inds0[cv0:]
    trainset1 = shuffled_inds1[:cv1]
    testset1 = shuffled_inds1[cv1:]

    train_x = torch.cat((buffer0[trainset0], buffer1[trainset1]), 0)
    train_y = []
    for j in range(cv0):
        train_y.append(0)
    for j in range(cv1):
        train_y.append(1)
    train_y = torch.tensor(train_y)

    test_x = torch.cat((buffer0[testset0], buffer1[testset1]), 0)
    test_y = []
    for j in range(len(buffer0)-cv0):
        test_y.append(0)
    for j in range(len(buffer1)-cv1):
        test_y.append(1)
    test_y = torch.tensor(test_y)

    with gpytorch.settings.max_cholesky_size(2000):

        #model1, mll1 = make_and_fit_classifier_s(train_x, train_y, oldproj0, oldprolen0, oldardlen0, mean0,
         #                                     lr=0.1)
        model1, mll1 = make_and_fit_classifier(train_x, train_y, inbuffer=basis,
                                               lr=0.05)
        testacc, trainacc, acc0, acc1 = compute_accuracy(model1, model1.likelihood, test_x, test_y,
                                                                      train_x, train_y)
        rmatrix, smatrix, proj, prolen, ardlen, mmean = cluster_lengthscales(model1)
    summatrix = summatrix + smatrix
    rawmatrix = rawmatrix + rmatrix
    proj_list.append(proj)
    prolen_list.append(prolen)
    ardlen_list.append(ardlen)
    mean_list.append(mmean)
    # r1=r1+r11
    acc_list.append(testacc)
    acc_list2.append(trainacc)
    acc0_list.append(acc0)
    acc1_list.append(acc1)

    mll_list.append(mll1)
    state_dict_list.append(model1.state_dict)

summatrix1 = summatrix.cpu()
print(summatrix1.shape)
for i in range(len(summatrix1)):
    for j in range(len(summatrix1)):
        if j <= i:
            summatrix1[i][j] = 0.
fig = plt.figure(figsize=(6, 6))
ax = plt.subplot()
f = plt.imshow(summatrix1.data.div(30.), cmap=mpl.cm.bwr, vmin=-1., vmax=1.)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.gca().invert_yaxis()
plt.xticks([0, 1],
           ['en2', 'fc3'],
           rotation=0)
plt.yticks([0, 1],
           ['en2', 'fc3'],
           rotation=0)
plt.colorbar(fraction=0.045)
plt.savefig(str(2) + "M_z_1.png", bbox_inches="tight", transparent='true')


oldproj0 = np.mean(proj_list, axis=0)
oldproj0 = torch.tensor(oldproj0, dtype=torch.float64).to(device)
oldprolen0 = np.mean(prolen_list, axis=0)
oldprolen0 = torch.tensor(oldprolen0, dtype=torch.float64).to(device)
oldardlen0 = np.mean(ardlen_list, axis=0)
oldardlen0 = torch.tensor(oldardlen0, dtype=torch.float64).to(device)
mean0 = np.mean(mean_list, axis=0)
mean0 = torch.tensor(mean0, dtype=torch.float64).to(device)


allx=torch.cat((buffer0,buffer1),0)
ally=[]
for i in range(len(buffer0)):
    ally.append(0)
for i in range(len(buffer1)):
    ally.append(1)
ally=torch.tensor(ally)
print(len(allx),len(ally))
likelihood = DirichletClassificationLikelihood(ally, learn_additional_noise=True)
model = DirichletGPModel(
    allx, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes,
    initialization=oldproj0,
    rank=2, interval1=100.
)
model.covar_module_projection.lengthscale = oldprolen0
model.covar_module_ard.lengthscale = oldardlen0
with torch.no_grad():
    model.mean_module.constant[0][0] = mean0[0][0]
    model.mean_module.constant[1][0] = mean0[1][0]
    model = model.to(allx)

model.eval()
likelihood.eval()







#unfix1=1
#unfix2=3

excel1 = xlrd.open_workbook(r"atomic.xls")
all_sheet1 = excel1.sheets()
atomicdata = all_sheet1[0]
ele1 = atomicdata.col_values(0)
elene1 = atomicdata.col_values(1)
elena1 = atomicdata.col_values(2)
eleip1 = atomicdata.col_values(3)
rcov1 = atomicdata.col_values(4)
excel2 = xlrd.open_workbook(r"xenonpy.xls")
all_sheet2 = excel2.sheets()
xenondata = all_sheet2[0]
polar = xenondata.col_values(57)
fccl = xenondata.col_values(25)
excel3 = xlrd.open_workbook(r"Econfig.xls")
all_sheet3 = excel3.sheets()
edata = all_sheet3[0]
ele2 = edata.col_values(0)[1:]
valence = edata.col_values(1)[1:]

fixdata = fixed_data(group)
#fixdsq = fixdata[1]
#fixdnn = fixdata[1]
prob=[]
for i in range(84):
    en=elene1[i]
    for k in range(len(ele2)):
        if ele2[k]==ele1[i]:
            fcc=fccl[k]
    point=torch.tensor([[en,fcc]], dtype=torch.float32)
    point = point.to(device)
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        test_dist = model(point)
        pred_samples = test_dist.sample(torch.Size((256,))).exp()
        probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)
        std_probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).std(0)
        pred_means = test_dist.loc
        pred_means = pred_means.cpu()
        prob.append(probabilities[1])


workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('sheet1')
for i in range(84):
    worksheet.write(i,0,label=ele1[i])
    worksheet.write(i,1,label=prob[i].cpu().data.numpy()[0])

workbook.save('element_prob.xls')



