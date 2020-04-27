#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sherpa.astro import ui
from bxa.sherpa.background.pca import PCAFitter, PCAModel, get_identity_response
import json


def fitBkgPCA(id=1):
    """
    The source model should be specified.
    """
    ui.set_stat('cash')
    bkgmodel = PCAFitter(id=id)  # An id has to be assigned.
    bkgmodel.fit()


def saveBkgPCA(id=1, writeTo='bkgPCA.json'):
    """
    Save the best-fit background model to a .json file.
    """
    bkgModel = ui.get_bkg_model(id=id)
    parDict = {p.fullname: p.val for p in bkgModel.pars}
    with open(writeTo, 'w') as f:
        json.dump(parDict, f)


def loadBkgPCA(id=1, readFrom='bkgPCA.json'):
    """
    load PCA background from
    """
    with open(readFrom, 'r') as f:
        parDict = json.load(f)
    fitter = PCAFitter(id=id)
    bkgModel = PCAModel('pca{:d}'.format(id), data=fitter.pca)
    for p in bkgModel.pars:
        p.val = parDict[p.fullname]
    idrsp = get_identity_response(1)
    ui.set_bkg_full_model(idrsp(bkgModel))


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
