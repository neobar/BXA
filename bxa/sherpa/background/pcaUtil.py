#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sherpa.astro import ui
from bxa.sherpa.background.pca import PCAFitter, PCAModel, get_identity_response
import json
from sherpautils.fitUtil import cstatDist


def fitBkgPCA(id=1):
    """
    The source model should be specified.

    Known Issues:
        While the photon indexes resulting from Wstat and PCA-based background fitting are consistent
        for MOS1 and MOS2, the photon index resulting from PCA-based background fitting for pn
        seems to be biased (steeper) by 0.04-0.05. The PCA-based model overestimates
        the background in the hard band.
    """
    bkgmodel = PCAFitter(id=id)  # An id has to be assigned.
    bkgmodel.fit()
    ui.set_analysis('channel')
    ui.ignore()
    ui.notice(bkgmodel.filter_chan)
    ui.set_analysis('energy')
    m, sig = cstatDist(id=id, bkg_id=1)
    bkgmodel.cstatM = m
    bkgmodel.cstatS = sig
    ui.set_analysis('channel')
    ui.ignore()
    ui.notice(bkgmodel.filter0)
    ui.set_analysis('energy')
    return bkgmodel


def saveBkgPCA(bkgFitter, id=1, writeTo=None, stat=True):
    """
    Save the best-fit background model to a .json file.
    bkgFitter, the instance of PCAFitter() should be given.
    Paramaters:
    id = 1
        The identification number of the dataset, the model of which is to be saved.
        Thefore, this function is not for joint fitting now.
    stat = True,
        backward compatibility, no effect.
    """
    bkgModel = ui.get_bkg_model(id=id)
    parDict = {p.fullname: p.val for p in bkgModel.pars}
    for i in ['cstat', 'dof', 'cstatM', 'cstatS', 'filter_chan']:
        parDict[i] = getattr(bkgFitter, i)
    if writeTo is not None:
        with open(writeTo, 'w') as f:
            json.dump(parDict, f, indent=4)
    return parDict


def saveSrcModel(id=1, writeTo='srcPowerLaw.json', stat=True, info={}):
    """
    """
    srcModel = ui.get_model(id=id)
    parDict = {p.fullname: p.val for p in srcModel.pars}
    if stat:
        fsrc, *_ = ui.get_stat_info()
        for i in ['statname', 'numpoints', 'dof', 'qval', 'rstat', 'statval']:
            parDict[i] = getattr(fsrc, i)
    for key, val in info.items():
        parDict[key] = val
    with open(writeTo, 'w') as f:
        json.dump(parDict, f, indent=4)


def saveModel(amodel, writeTo, stat=True, info={}):
    """
    Save the model paramaters to writeTo file.
    Paramaters
    stat = True
        if stat is True, save the stat info as well.
    """
    # The values of Dict is a pair of (value, frozen).
    # The paramater is frozen if frozen is True.
    parDict = {p.name: (p.val, p.frozen) for p in amodel.pars}
    if stat:
        fit = ui.get_fit_results()  # the results of the last fit
        for i in ['statname', 'numpoints', 'dof', 'qval', 'rstat', 'statval']:
            parDict[i] = getattr(fit, i)
    for key, val in info.items():
        parDict[key] = val
    with open(writeTo, 'w') as f:
        json.dump(parDict, f, indent=4)


def loadPars(amodel, readFrom):
    """
    """
    with open(readFrom, 'r') as f:
        parDict = json.load(f)
    for p in amodel.pars:
        val, frozen = parDict[p.name]
        p.val = val
        if frozen:
            ui.freeze(p)


def loadSrcModel(id=1, readFrom='srcPowerLaw.json'):
    """
    """
    with open(readFrom, 'r') as f:
        parDict = json.load(f)
    srcModel = ui.get_model(id=id)
    for p in srcModel.pars:
        p.val = parDict[p.fullname]


def loadBkgPCA(id=1, readFrom='bkgPCA.json'):
    """
    load PCA background from
    """
    filter0 = ui.get_filter()

    with open(readFrom, 'r') as f:
        parDict = json.load(f)
        parDictRed = {k.split('.')[1]: v for k, v in parDict.items() if k.startswith('pca')}
    fitter = PCAFitter(id=id)
    bkgModel = PCAModel('pca{:d}'.format(id), data=fitter.pca)
    for p in bkgModel.pars:
        p.val = parDictRed[p.name]
    idrsp = get_identity_response(id)
    ui.set_bkg_full_model(id, idrsp(bkgModel))
    ui.ignore()
    ui.notice(filter0)


def fixBkgModel(id=1, freeNorm=True):
    """
    Fix the background model
    Paramaters:
    freeNorm = True
        If freeNorm is True, leave the overall background normalization free.
    """
    bk = ui.get_bkg_model(id=id)
    if freeNorm:
        pars = bk.pars
        c0 = pars[0]
        for p in pars[1:]:
            if p.val != 0 and p.fullname.startswith('pca'):
                ui.link(p, p.val / c0.val * c0)
            else:
                ui.freeze(p)
    else:
        ui.freeze(bk)
