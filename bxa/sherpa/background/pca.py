#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import json
import logging
import os

import sherpa.astro.ui as ui
from sherpa.models.parameter import Parameter
from sherpa.models import ArithmeticModel, CompositeModel
from sherpa.astro.instrument import RSPModelNoPHA, RMFModelNoPHA


def pca(M):
    mean = M.mean(axis=0)
    Moffset = M - mean.reshape((1, -1))
    U, s, Vt = np.linalg.svd(Moffset, full_matrices=False)
    V = Vt.T
    print('variance explained:', s**2/len(M))
    c = np.cumsum(s**2/len(M))
    c = c / c[-1]
    for cut in 0.80, 0.90, 0.95, 0.99:
        idx, = np.where(c > cut)
        n = idx.min() + 1
        print('  --> need %d components to explain %.2f%%' % (n, cut))
    return U, s, V, mean


def pca_predict(U, s, V, mean):
    S = np.diag(s)
    return np.dot(U, np.dot(S, V.T)) + mean.reshape((1, -1))


def pca_get_vectors(s, V, mean):
    # U = np.eye(len(s))
    # return pca_predict(U, s, V, mean)
    Sroot = np.diag(s**0.5)
    return np.dot(Sroot, V.T)


def pca_cut(U, s, V, mean, ncomponents=20):
    return U[:, :ncomponents], s[:ncomponents], V[:, :ncomponents], mean


def pca_check(M, U, s, V, mean):
    # if we use only the first 20 PCs the reconstruction is less accurate
    Mhat2 = pca_predict(U, s, V, mean)
    print("Using %d PCs, MSE = %.6G" % (len(s), np.mean((M - Mhat2)**2)))
    return M - Mhat2


class IdentityPileupResponse(CompositeModel, ArithmeticModel):
    """
    When the piledup model (jdpileup) is used.
    """
    def __init__(self, n, model, rmf, arf, pha):
        self.n = n
        self.elo = np.arange(n)
        self.ehi = np.arange(n)
        self.lo = np.arange(n)
        self.hi = np.arange(n)
        self.xlo = np.arange(n)
        self.xhi = np.arange(n)

        self.pha = pha
        self.channel = pha.get_noticed_channels()
        self.mask = pha.get_mask()

        self.rmf = rmf
        self.arf = arf
        self.model = model
        CompositeModel.__init__(self, ('%s(%s)' % ('apply_identity_rsp', self.model.name)), (model,))

    def startup(self, *args):
        self.model.startup(*args)
        CompositeModel.startup(self, *args)

    def apply_rmf(self, src):
        return src

    def teardown(self):
        self.model.teardown()
        CompositeModel.teardown(self)

    def calc(self, p, x, xhi=None, **kwargs):
        vals = self.model.calc(p, self.xlo, self.xhi)
        assert np.isfinite(vals).all(), vals
        if self.mask is not None:
            vals = vals[self.mask]
        return vals


class IdentityResponse(RSPModelNoPHA):
    def __init__(self, n, model, arf, rmf):
        self.n = n
        RSPModelNoPHA.__init__(self, arf=arf, rmf=rmf, model=model)
        self.elo = np.arange(n)
        self.ehi = np.arange(n)
        self.lo = np.arange(n)
        self.hi = np.arange(n)
        self.xlo = np.arange(n)
        self.xhi = np.arange(n)

    def apply_rmf(self, src):
        return src

    def calc(self, p, x, xhi=None, *args, **kwargs):
        src = self.model.calc(p, self.xlo, self.xhi)
        assert np.isfinite(src).all(), src
        return src


class IdentityRMF(RMFModelNoPHA):
    def __init__(self, n, model, rmf):
        self.n = n
        RMFModelNoPHA.__init__(self, rmf=rmf, model=model)
        self.elo = np.arange(n)
        self.ehi = np.arange(n)
        self.lo = np.arange(n)
        self.hi = np.arange(n)
        self.xlo = np.arange(n)
        self.xhi = np.arange(n)

    def apply_rmf(self, src):
        return src

    def calc(self, p, x, xhi=None, *args, **kwargs):
        src = self.model.calc(p, self.xlo, self.xhi)
        assert np.isfinite(src).all(), src
        return src


def replace_bkg_identity_response(i=1):
    """
    The PileupRMFModel(), by default, only calculate convolved model at noticed channel.
    See https://github.com/sherpa/sherpa/blob/master/sherpa/astro/instrument.py

    Here, simply replace the response of the background that calculates at all channels to
    that caclculate background component at only noticed channel.
    """
    pha = ui.get_bkg(i)
    n = pha.counts.size
    src = ui.get_data(i)  # mask background according to src.
    bkgModel = ui.get_bkg_model().model
    rmf = ui.get_bkg_rmf(i)
    arf = ui.get_bkg_arf(i)
    return IdentityPileupResponse(n, bkgModel, rmf=rmf, arf=arf, pha=src)


def get_identity_response(i):
    n = ui.get_bkg(i).counts.size
    rmf = ui.get_rmf(i)
    try:
        arf = ui.get_arf(i)
        return lambda model: IdentityResponse(n, model, arf=arf, rmf=rmf)
    except:
        return lambda model: IdentityRMF(n, model, rmf=rmf)


logf = logging.getLogger('bxa.Fitter')
logf.setLevel(logging.INFO)


class PCAModel(ArithmeticModel):  # Model
    def __init__(self, modelname, data):
        # self.U = data['U']
        self.V = np.matrix(data['components'])
        self.mean = data['mean']
        self.s = data['values']
        self.ilo = data['ilo']
        self.ihi = data['ihi']

        p0 = Parameter(modelname=modelname, name='lognorm', val=1, min=-5, max=20,
                       hard_min=-100, hard_max=100)
        pars = [p0]
        for i in range(len(self.s)):
            pi = Parameter(modelname=modelname, name='PC%d' % (i+1),
                           val=0, min=-20, max=20,
                           hard_min=-1e300, hard_max=1e300)
            pars.append(pi)
        super(ArithmeticModel, self).__init__(modelname, pars=pars)

    def calc(self, p, left, right, *args, **kwargs):
        try:
            lognorm = p[0]
            pars = np.array(p[1:])
            y = np.array(pars * self.V.T + self.mean).flatten()
            cts = (10**y - 1) * 10**lognorm
            cts[cts <= 0] = 1E-15  # prevent model from predicting negative values.
            out = np.zeros_like(left, dtype=float)  # out = left * 0.0
            out[self.ilo:self.ihi] = cts
            return out
        except Exception as e:
            print("Exception in PCA model:", e, p)
            raise e

    def startup(self, *args):
        pass

    def teardown(self, *args):
        pass

    def guess(self, dep, *args, **kwargs):
        self._load_params()


class GaussModel(ArithmeticModel):
    def __init__(self, modelname):
        self.LineE = Parameter(modelname=modelname, name='LineE', val=1, min=0, max=1e38)
        self.Sigma = Parameter(modelname=modelname, name='Sigma', val=1, min=0, max=1e38)
        self.norm = Parameter(modelname=modelname, name='norm', val=1, min=0, max=1e38)
        pars = (self.LineE, self.Sigma, self.norm)
        super(ArithmeticModel, self).__init__(modelname, pars=pars)

    def calc(self, p, left, right, *args, **kwargs):
        try:
            LineE, Sigma, norm = p
            cts = norm * np.exp(-0.5 * ((left - LineE)/Sigma)**2)
            return cts
        except Exception as e:
            print("Exception in PCA model:", e, p)
            raise e

    def startup(self, cache):
        pass

    def teardown(self):
        pass

    def guess(self, dep, *args, **kwargs):
        self._load_params()


class PCAFitter(object):
    def __init__(self, id=None, bkgModelDir=None):
        """
        Find background model.

        id: which data id to fit
        bkgModelDir: read background model files from a different directories.

        I analysed the background for many instruments, and stored mean and
        principle components. The data file tells us which instrument we deal with,
        so we load the correct file.
        First guess:
        1) PCA decomposition.
        2) Mean scaled, other components zero
        The one with the better cstat is kept.
        Then start with 0 components and add 1, 2 components until no improvement
        in AIC/cstat.
        """
        self.id = id
        bkg = ui.get_bkg(self.id)
        hdr = bkg.header
        telescope = hdr.get('TELESCOP', '')
        instrument = hdr.get('INSTRUME', '')
        if telescope == '' and instrument == '':
            raise Exception('ERROR: The TELESCOP/INSTRUME headers are not set in the data file.')
        self.data = bkg.counts
        self.ndata = len(self.data)
        self.ngaussians = 0
        if bkgModelDir is None:
            bkgModelDir = os.path.dirname(__file__)
        style1 = os.path.join(bkgModelDir, ('%s_%s_%d.json' % (telescope, instrument, self.ndata)).lower())
        style2 = os.path.join(bkgModelDir, ('%s_%d.json' % (telescope, self.ndata)).lower())
        if os.path.exists(style1):
            self.load(style1)
        elif os.path.exists(style2):
            self.load(style2)
        else:
            raise Exception('ERROR: Could not load PCA components for this detector (%s %s, %d channels). Try the SingleFitter instead.' % (telescope, instrument, self.ndata))

    def load(self, filename):
        self.modelFile = filename
        with open(filename, 'r') as f:
            self.pca = json.load(f)
        for k, v in self.pca.items():
            self.pca[k] = np.array(v)
        nactivedata = self.pca['ihi'] - self.pca['ilo']
        assert self.pca['hi'].shape == (nactivedata,), 'spectrum has different number of channels: %d vs %s' % (len(self.pca['hi']), self.ndata)
        assert self.pca['lo'].shape == self.pca['hi'].shape
        assert self.pca['mean'].shape == self.pca['hi'].shape
        assert len(self.pca['components']) == nactivedata
        assert nactivedata <= self.ndata
        ilo = int(self.pca['ilo'])
        ihi = int(self.pca['ihi'])
        self.cts = self.data[ilo:ihi]
        self.ncts = self.cts.sum()  # 'have ncts background counts for deconvolution
        self.x = np.arange(ihi-ilo)
        self.ilo = ilo
        self.ihi = ihi

        # Only notice the channels between ilo + 1 and ihi (channel starts from 1, while index from 0).
        # The stat value will be affected, for assessment of goodness-of-fit for background.
        ui.set_analysis('channel')
        self.filter0 = ui.get_filter()
        ui.ignore()
        ui.notice(self.ilo + 1, self.ihi)  # ui.notice(a, b), from channel a to channel b, including channels a, b.
        ui.set_analysis('energy')

    def decompose(self):
        mean = self.pca['mean']
        V = np.matrix(self.pca['components'])
        s = self.pca['values']
        y = np.log10(self.cts * 1. / self.ncts + 1.0)
        z = (y - mean) * V
        assert z.shape == (1, len(s)), z.shape
        z = z.tolist()[0]
        return np.array([np.log10(self.ncts + 0.1)] + z)

    def calc_bkg_stat(self, dof=False):
        ss = [s for s in ui.get_stat_info() if self.id in s.ids and s.bkg_ids is not None and len(s.bkg_ids) > 0]
        if len(ss) != 1:
            for s in ui.get_stat_info():
                if self.id in s.ids and len(s.bkg_ids) > 0:
                    print('get_stat_info returned: ids=%s bkg_ids=%s' % (s.ids, s.bkg_ids))
        assert len(ss) == 1
        return (ss[0].statval, ss[0].dof) if dof else ss[0].statval

    def fit(self):
        # try a PCA decomposition of this spectrum
        initial = self.decompose()
        ui.set_method('neldermead')
        bkgmodel = PCAModel('pca%s' % self.id, data=self.pca)
        self.bkgmodel = bkgmodel
        response = get_identity_response(self.id)
        convbkgmodel = response(bkgmodel)
        ui.set_bkg_full_model(self.id, convbkgmodel)
        for p, v in zip(bkgmodel.pars, initial):
            p.val = v
        srcmodel = ui.get_model(self.id)
        ui.set_full_model(self.id, srcmodel)
        initial_v = self.calc_bkg_stat()
        # print('before fit: stat: %s' % (initial_v))
        ui.fit_bkg(id=self.id)
        # print('fit: first full fit done')
        final = [p.val for p in ui.get_bkg_model(self.id).pars]
        # print('fit: parameters: %s' % (final))
        initial_v = self.calc_bkg_stat()
        # print('fit: stat: %s' % (initial_v))

        # lets try from zero
        # logf.info('fit: second full fit from zero')
        for p in bkgmodel.pars:
            p.val = 0
        ui.fit_bkg(id=self.id)
        initial_v0 = self.calc_bkg_stat()
        # logf.info('fit: parameters: %s' % (final))
        # logf.info('fit: stat: %s' % (initial_v0))

        # pick the better starting point
        if initial_v0 < initial_v:
            # logf.info('fit: using zero-fit')
            initial_v = initial_v0
            final = [p.val for p in ui.get_bkg_model(self.id).pars]
        else:
            # logf.info('fit: using decomposed-fit')
            for p, v in zip(bkgmodel.pars, final):
                p.val = v

        # start with the full fit and remove(freeze) parameters
        print('%d parameters, stat=%.2f' % (len(initial), initial_v))
        results = [(2 * len(final) + initial_v, final, len(final), initial_v)]
        for i in range(len(initial)-1, 0, -1):
            bkgmodel.pars[i].val = 0
            bkgmodel.pars[i].freeze()
            ui.fit_bkg(id=self.id)
            final = [p.val for p in ui.get_bkg_model(self.id).pars]
            v = self.calc_bkg_stat()
            print('--> %d parameters, stat=%.2f' % (i, v))
            results.insert(0, (v + 2*i, final, i, v))

        print()
        print('Background PCA fitting AIC results:')
        print('-----------------------------------')
        print()
        print('stat Ncomp AIC')
        for aic, params, nparams, val in results:
            print('%-05.1f %2d %-05.1f' % (val, nparams, aic))
        aic, final, nparams, val = min(results)
        for p, v in zip(bkgmodel.pars, final):
            p.val = v
        for i in range(nparams):
            bkgmodel.pars[i].thaw()

        print()
        print('Increasing parameters again...')
        # now increase the number of parameters again
        # results = [(aic, final, nparams, val)]
        last_aic, last_final, last_nparams, last_val = aic, final, nparams, val
        for i in range(last_nparams, len(bkgmodel.pars)):
            next_nparams = i + 1
            bkgmodel.pars[i].thaw()
            for p, v in zip(bkgmodel.pars, last_final):
                p.val = v
            ui.fit_bkg(id=self.id)
            next_final = [p.val for p in ui.get_bkg_model(self.id).pars]
            v = self.calc_bkg_stat()
            next_aic = v + 2*next_nparams
            if next_aic < last_aic:  # accept
                print('%d parameters, aic=%.2f ** accepting' % (next_nparams, next_aic))
                last_aic, last_final, last_nparams, last_val = next_aic, next_final, next_nparams, v
            else:
                print('%d parameters, aic=%.2f' % (next_nparams, next_aic))
            # stop if we are 3 parameters ahead what we needed
            if next_nparams >= last_nparams + 3:
                break

        print('Final choice: %d parameters, aic=%.2f' % (last_nparams, last_aic))
        # reset to the last good solution
        for p, v in zip(bkgmodel.pars, last_final):
            p.val = v

        last_model = convbkgmodel
        for i in range(10):
            print('Adding Gaussian#%d' % (i+1))
            # find largest discrepancy
            ui.set_analysis(self.id, "ener", "rate")
            m = ui.get_bkg_fit_plot(self.id)
            y = m.dataplot.y.cumsum()
            z = m.modelplot.y.cumsum()
            diff_rate = np.abs(y - z)
            ui.set_analysis(self.id, "ener", "counts")
            m = ui.get_bkg_fit_plot(self.id)
            x = m.dataplot.x
            y = m.dataplot.y.cumsum()
            z = m.modelplot.y.cumsum()
            diff = np.abs(y - z)
            i = np.argmax(diff)
            energies = x
            e = x[i]
            print('largest remaining discrepancy at %.3fkeV[%d], need %d counts' % (x[i], i, diff[i]))
            # e = x[i]
            power = diff_rate[i]
            # lets try to inject a gaussian there

            g = ui.xsgaussian('g_%d_%d' % (self.id, i))
            print('placing gaussian at %.2fkeV, with power %s' % (e, power))
            # we work in energy bins, not energy
            g.LineE.min = energies[0]
            g.LineE.max = energies[-1]
            g.LineE.val = e
            if i > len(diff) - 2:
                i = len(diff) - 2
            if i < 2:
                i = 2
            g.Sigma = (x[i + 1] - x[i - 1])
            g.Sigma.min = (x[i + 1] - x[i - 1])/3
            g.Sigma.max = x[-1] - x[0]
            g.norm.min = power * 1e-6
            g.norm.val = power
            convbkgmodel2 = response(g)
            next_model = last_model + convbkgmodel2
            ui.set_bkg_full_model(self.id, next_model)
            ui.fit_bkg(id=self.id)
            next_final = [p.val for p in ui.get_bkg_model(self.id).pars]
            next_nparams = len(next_final)
            v = self.calc_bkg_stat()
            next_aic = v + 2 * next_nparams
            print('with Gaussian:', next_aic, '; change: %.1f (negative is good)' % (next_aic - last_aic))
            if next_aic < last_aic:
                print('accepting')
                last_model = next_model
                last_aic, last_final, last_nparams, last_val = next_aic, next_final, next_nparams, v
            else:
                print('not significant, rejecting')
                ui.set_bkg_full_model(self.id, last_model)
                for p, v in zip(last_model.pars, last_final):
                    p.val = v
                    if v == 0:  # the parameter was frozen.
                        ui.freeze(p)
                break

        self.cstat, self.dof = self.calc_bkg_stat(dof=True)  # Save the final cstat and dof (dof = ihi - ilo)
        self.filter_energy = ui.get_filter()  # Save the filter for background fitting.
        ui.set_analysis('channel')
        self.filter_chan = ui.get_filter()  # Save the filter for background fitting.
        ui.ignore()
        ui.notice(self.filter0)  # restore filter
        ui.set_analysis('energy')


__dir__ = [PCAFitter, PCAModel]
