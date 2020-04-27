#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sherpa.astro import ui
from bxa.sherpa.background.pca import PCAFitter


def fitBkgPCA(id=1):
    """
    The source model should be specified.
    """
    ui.set_stat('cash')
    bkgmodel = PCAFitter(id=id)  # An id has to be assigned.
    bkgmodel.fit()


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
