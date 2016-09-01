import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
import astropy.io.ascii as aio
import astropy.units as u
from astropy.analytic_functions import blackbody_nu

#
# Clase che definisce la singola stella
class Disk(object):

    def __init__(self, tabrow, full_table=True, tab_fmt='lupus'):
        #
        self.tabrow = tabrow
        self.star_id = self.tabrow['ID']
        self.star_name = self.tabrow['Name']
        self.dist = self.tabrow['Dist']
        self.teff = self.tabrow['Teff']
        self.lstar = self.tabrow['Lstar']
        self.llstar = np.log10(self.lstar)
        self.mstar = self.tabrow['Mstar']
        self.lmstar = np.log10(self.mstar)
        self.fcont = self.tabrow['F_cont']
        self.efcont = self.tabrow['E_cont']
        #
        if tab_fmt == 'lupus':
            self.read_marco_fit_params()
            self.read_lupus_ansdell_pars()
        if tab_fmt == 'bds':
            self.read_marco_fit_params()
            self.read_bds_pars()
            k_fact=1. #200./337.
            self.mdust = self.mdust*k_fact
            self.mdust_16 = self.mdust_16*k_fact
            self.mdust_84 = self.mdust_84*k_fact
            self.sigma0 = self.sigma0*k_fact
            self.sigma0_16 = self.sigma0_16*k_fact
            self.sigma0_84 = self.sigma0_84*k_fact
            self.mdust_testi = self.mdust_testi*k_fact
        self.compute_tmm()

    def compute_tmm(self):
        # Compute Tmm
        # pc = 3.0857e18 cm
        # mJy = 1.e-26  erg /s/cm2/Hz
        # mE = 5.9722e27 g
        # k = mJy*pc*pc/mE = 1.594311e-17
        self.mfunc_k = 1.594311e-17
        self.tmm_fs = self.get_tmm(self.fcont,self.mdust)
        self.tmm_fs_16 = self.get_tmm(self.fcont,self.mdust_16)
        self.tmm_fs_84 = self.get_tmm(self.fcont,self.mdust_84)
        self.tmm = self.tmm_fs[0]
        self.tmm_16 = self.tmm_fs_16[0]
        self.tmm_84 = self.tmm_fs_84[0]

    def read_marco_fit_params(self):
        self.rout = self.tabrow['R_out_50']
        self.rout_16 = self.tabrow['R_out_16']
        self.rout_84 = self.tabrow['R_out_84']
        self.mdust = self.tabrow['M_dust_50']
        self.mdust_16 = self.tabrow['M_dust_16']
        self.mdust_84 = self.tabrow['M_dust_84']
        self.gamma = self.tabrow['gamma_50']
        self.gamma_16 = self.tabrow['gamma_16']
        self.gamma_84 = self.tabrow['gamma_84']
        self.rc = self.tabrow['Rc_50']
        self.rc_16 = self.tabrow['Rc_16']
        self.rc_84 = self.tabrow['Rc_84']
        self.sigma0 = self.tabrow['sigma0_50']
        self.sigma0_16 = self.tabrow['sigma0_16']
        self.sigma0_84 = self.tabrow['sigma0_84']
        self.inc = self.tabrow['inc_50']
        self.inc_16 = self.tabrow['inc_16']
        self.inc_84 = self.tabrow['inc_84']
        if self.tabrow['Valid'] == 1:
            self.valid = True 
        else:
            self.valid = False        

    def read_lupus_ansdell_pars(self):
        self.i = self.tabrow['i']
        self.ei = self.tabrow['e_i']
        self.a = self.tabrow['a']
        self.ea = self.tabrow['e_a']
        self.mdust_ansdell = self.tabrow['M_dust']

    def read_bds_pars(self):
        self.a = self.tabrow['a']
        self.ea = self.tabrow['e_a']
        self.mdust_testi = self.tabrow['M_dust']

    def _my_p_fac(self,x):
        # planck factor
        xmin = 0.001
        xmax = 100.
        if x < xmin:
            return 1./(x+x*x/2.)
        elif x > xmax:
            return np.exp(-x)
        else:
            return 1./(np.exp(x)-1.0)
    def get_planck(self,t,n):
        #
        # nu in GHz -> h*1e9  c*1e-9
        # h = 6.626070040e-27
        hghz = 6.626070040e-18
        # kb = 1.38064852e-16
        hkbghz = 4.79924466e-2
        # c = 29979245800 cm/s
        cghz = 29.9792458
        
        c1 = 2.*hghz*n*n*n/cghz/cghz
        c2 = hkbghz*n/t
        
        f2 = self._my_p_fac(c2)
        
        return c1*f2

    def get_tmm(self, f890, md, kappa = 3.4, nu = 340.):
        #
        ft = lambda x: self.get_planck(x, nu) - self.mfunc_k*f890*self.dist*self.dist/kappa/md
        #ft = lambda x: (blackbody_nu(nu*u.GHz,x*u.K)).value() - self.mfunc_k*f890*self.dist*self.dist/kappa/md
        return so.fsolve(ft,20.)
    
#
# classe che definisce il sample di stelle
class DiskSample(object):
    
    def __init__(self, tabfile, tab_fmt='lupus'):
        #
        self.original_file = tabfile
        self.mytable = aio.read(tabfile,format='ipac',guess=False)
        self.stars = []
        [self.stars.append(Disk(row,tab_fmt=tab_fmt)) for row in self.mytable]
        self.nstars = len(self.stars)
        self.fill_err_cols()
        self.fill_tmm_cols()
        self.nval = np.where(self.mytable['Valid'] == 1 )
        self.ninval = np.where(self.mytable['Valid'] == 0 )
                
    def __str__(self):
        rep = "Table of Disk observed and fitted properties, read from file %s containing %d objects (%d validated)" % (self.original_file,self.nstars,len(self.nval[0]))
        return rep
    
    def fill_err_cols(self):  
        rom=[]
        rop=[]
        rcm=[]
        rcp=[]
        for i in range(self.nstars):
            if (self.mytable[i]['Rc_16'] > self.mytable[i]['Rc_84']):
                rcp.append(self.mytable[i]['Rc_16']-self.mytable[i]['Rc_50'])
                rcm.append(self.mytable[i]['Rc_50']-self.mytable[i]['Rc_84'])
            else:
                rcm.append(-self.mytable[i]['Rc_16']+self.mytable[i]['Rc_50'])
                rcp.append(-self.mytable[i]['Rc_50']+self.mytable[i]['Rc_84'])
        for i in range(self.nstars):
            if (self.mytable[i]['R_out_16'] > self.mytable[i]['R_out_84']):
                rop.append(self.mytable[i]['R_out_16']-self.mytable[i]['R_out_50'])
                rom.append(self.mytable[i]['R_out_50']-self.mytable[i]['R_out_84'])
            else:
                rom.append(-self.mytable[i]['R_out_16']+self.mytable[i]['R_out_50'])
                rop.append(-self.mytable[i]['R_out_50']+self.mytable[i]['R_out_84'])
        self.mytable['R_out_ep'] = rop
        self.mytable['R_out_em'] = rom
        self.mytable['Rc_ep'] = rcp
        self.mytable['Rc_em'] = rcm

    def fill_tmm_cols(self):
        a=[]
        b=[]
        c=[]
        d=[]
        e=[]
        for i in range(self.nstars):
            a.append(self.stars[i].tmm)
            b.append(self.stars[i].tmm_16)
            c.append(self.stars[i].tmm_84)
            if (self.stars[i].tmm_16 > self.stars[i].tmm_84):
                d.append(self.stars[i].tmm_16-self.stars[i].tmm)
                e.append(self.stars[i].tmm-self.stars[i].tmm_84)
            else:
                e.append(-self.stars[i].tmm_16+self.stars[i].tmm)
                d.append(-self.stars[i].tmm+self.stars[i].tmm_84)
        self.mytable['Tmm_16'] = b * u.K
        self.mytable['Tmm_50'] = a * u.K
        self.mytable['Tmm_84'] = c * u.K
        self.mytable['Tmm_ep'] = d * u.K
        self.mytable['Tmm_em'] = e * u.K

    def write_table(self, outfile):
        self.mytable.write(outfile,format='ascii.ipac')

    #
    # Computes the Tlt temperature
    def calc_tlt(self,tmed):
        #
        self.mytable['Tlt'] = tmed*(self.mytable['Lstar']/self.mytable['Mstar'])**0.25 * u.K

    #
    # Computes the Tlt_Ruvg temperature
    def calc_tlt_uvg(self,tmed,amin):
        # amin is in arcsec
        #
        self.mytable['Tlt_uvg'] = self.mytable['Tmm_50']
        for i in range(len(self.mytable['Lstar'])):
            if self.mytable['a'][i] != '...':
                #print('{0}, {1}, {2}, {3}'.format(tmed,self.mytable['Lstar'][i],float(self.mytable['a'][i]),self.mytable['Dist'][i]))
                #print('{0} = {1}'.format(self.mytable['Tlt_uvg'][i],tmed*(self.mytable['Lstar'][i])**0.25/(float(self.mytable['a'][i])*self.mytable['Dist'][i])**0.5 * u.K))
                self.mytable['Tlt_uvg'][i] = tmed*(self.mytable['Lstar'][i])**0.25/(float(self.mytable['a'][i])*self.mytable['Dist'][i])**0.5 #* u.K
            else:
                self.mytable['Tlt_uvg'][i] = tmed*(self.mytable['Lstar'][i])**0.25/(amin*self.mytable['Dist'][i])**0.5 #* u.K
        self.mytable['Tlt_uvg'] = self.mytable['Tlt_uvg'] * u.K

    #
    # Computes the Tlt_LaM temperature
    def calc_tlt_lam(self,tmed,amin):
        # amin is in arcsec
        #
        self.mytable['Tlt_lam'] = self.mytable['Tmm_50']
        for i in range(len(self.mytable['Lstar'])):
            if self.mytable['a'][i] != '...':
                #print('{0}, {1}, {2}, {3}'.format(tmed,self.mytable['Lstar'][i],float(self.mytable['a'][i]),self.mytable['Dist'][i]))
                #print('{0} = {1}'.format(self.mytable['Tlt_uvg'][i],tmed*(self.mytable['Lstar'][i])**0.25/(float(self.mytable['a'][i])*self.mytable['Dist'][i])**0.5 * u.K))
                self.mytable['Tlt_lam'][i] = tmed*(self.mytable['Lstar'][i]/self.mytable['Mstar'][i])**0.25/(float(self.mytable['a'][i])*self.mytable['Dist'][i])**0.5 #* u.K
            else:
                self.mytable['Tlt_lam'][i] = tmed*(self.mytable['Lstar'][i]/self.mytable['Mstar'][i])**0.25/(amin*self.mytable['Dist'][i])**0.5 #* u.K
        self.mytable['Tlt_lam'] = self.mytable['Tlt_lam'] * u.K

    #
    # Computes the Tlt temperature
    def calc_ta(self):
        #
        self.mytable['Ta'] = 25.*(self.mytable['Lstar'])**0.25 * u.K
        self.mytable['Tvdp'] = 22.*(self.mytable['Lstar'])**0.16 * u.K

    def get_mass(self,bfac):
        # pc = 3.0857e18 cm
        # mJy = 1.e-26  erg /s/cm2/Hz
        # mE = 5.9722e27 g
        # k340 =3.4
        # k = mJy*pc*pc/mE = 1.594311e-17
        mfunc_k = 1.594311e-17/self.k340
        m = mfunc_k * self.mytable['F_cont'] * self.mytable['Dist'] * self.mytable['Dist'] / bfac
        m_ep = mfunc_k * (self.mytable['F_cont']+self.mytable['E_cont']) * self.mytable['Dist'] * self.mytable['Dist'] / bfac - m
        m_em = m - mfunc_k * (self.mytable['F_cont']-self.mytable['E_cont']) * self.mytable['Dist'] * self.mytable['Dist'] / bfac
        return m, m_ep, m_em

    #
    # Computes the md (and uncertainties) using the Tlt, Ta temperatures 
    def calc_md_tlt(self, nu = 340.*u.GHz, k340 = 3.37):
        self.k340 = k340
        self.boltz_tlt = np.ones(self.nstars)
        self.boltz_tlt_uvg = np.ones(self.nstars)
        self.boltz_tlt_lam = np.ones(self.nstars)
        self.boltz_ta = np.ones(self.nstars)
        self.boltz_tvdp = np.ones(self.nstars)
        for i in range(self.nstars):
            self.boltz_tlt[i] = blackbody_nu(nu, self.mytable[i]['Tlt']).value
            self.boltz_tlt_uvg[i] = blackbody_nu(nu, self.mytable[i]['Tlt_uvg']).value
            self.boltz_tlt_lam[i] = blackbody_nu(nu, self.mytable[i]['Tlt_lam']).value
            self.boltz_ta[i] = blackbody_nu(nu, self.mytable[i]['Ta']).value
            self.boltz_tvdp[i] = blackbody_nu(nu, self.mytable[i]['Tvdp']).value
        self.boltz_20k = np.zeros(self.nstars)+blackbody_nu(nu, 20. * u.K).value
        
        #
        self.mlt, self.mlt_ep, self.mlt_em = self.get_mass(self.boltz_tlt)
        self.mlt_uvg, self.mlt_uvg_ep, self.mlt_uvg_em = self.get_mass(self.boltz_tlt_uvg)
        self.mlt_lam, self.mlt_lam_ep, self.mlt_lam_em = self.get_mass(self.boltz_tlt_lam)
        self.mta, self.mta_ep, self.mta_em = self.get_mass(self.boltz_ta)
        self.mtvdp, self.mtvdp_ep, self.mtvdp_em = self.get_mass(self.boltz_tvdp)
        self.m20k, self.m20k_ep, self.m20k_em = self.get_mass(self.boltz_20k)

    def do_LM_plot(self, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, newfig=True, 
                   fsiz=(8,6), f='None', myyrange=[2.,1000.], myxrange=[0.03,4.], dolabel=True):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            f = plt.figure(figsize=fsiz)

        myxlm = (self.mytable['Lstar']/self.mytable['Mstar'])**0.25

        # Plot for Rout
        plt.errorbar(self.mytable[self.nval]['Mstar'],self.mytable[self.nval]['Tmm_50']/myxlm[self.nval],
             yerr=[self.mytable[self.nval]['Tmm_em']/myxlm[self.nval],self.mytable[self.nval]['Tmm_ep']/myxlm[self.nval]], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        plt.errorbar(self.mytable[self.ninval]['Mstar'],self.mytable[self.ninval]['Tmm_50']/myxlm[self.ninval],
             yerr=[self.mytable[self.ninval]['Tmm_em']/myxlm[self.ninval],self.mytable[self.ninval]['Tmm_ep']/myxlm[self.ninval]], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)

        # compute median, plot and annotate
        self.Tlt = np.median(self.mytable[self.nval]['Tmm_50']/myxlm[self.nval])
        plt.plot(myxrange,[self.Tlt,self.Tlt],linestyle='dashed',color=mycolor)
        xl = myxrange[0]+(myxrange[1]-myxrange[0])*0.005
        yl = myyrange[1]-(myyrange[1]-myyrange[0])*0.2
        mylab = "Median = %5.2f" % (self.Tlt)
        if dolabel:
            plt.text(xl,yl,mylab)
        
        # Y-axis
        plt.xscale('log')
        #ax[0].set_xlabel(r'(L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$')
        plt.xlabel(r'M$_\star$/M$_\odot$')
        plt.xlim(myxrange[0],myxrange[1])
        # Y-axis
        plt.ylabel(r'(T$_{mm}$/K)/((L$_\star$/L$_\odot$)/(M$_\star$/M$_\odot$))$^{0.25}$')
        plt.yscale('log')
        plt.ylim(myyrange[0],myyrange[1])

        # Plot for Rc
        #ax[1].errorbar(self.mytable[self.nval]['Mstar'],self.mytable[self.nval]['Rc_50']/self.mytable[self.nval]['Mstar']**0.5,
        #     yerr=[self.mytable[self.nval]['Rc_em'],self.mytable[self.nval]['Rc_ep']], 
        #     fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        #ax[1].errorbar(self.mytable[self.ninval]['Mstar'],self.mytable[self.ninval]['Rc_50']/self.mytable[self.ninval]['Mstar']**0.5,
        #     yerr=[self.mytable[self.ninval]['Rc_em'],self.mytable[self.ninval]['Rc_ep']], 
        #     mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        #ax[1].set_xscale('log')
        #ax[1].set_xlabel(r'M$_\star$/M$_\odot$')
        #ax[1].set_xlim(myxrange[0],myxrange[1])
        # Y-axis
        #ax[1].set_ylabel(r'(R$_c$/AU)/(M$_\star$/M$_\odot$)$^{0.5}$')
        #ax[1].set_yscale('log')
        #ax[1].set_ylim(myyrange[0],myyrange[1])

        if newfig:
            return f
        else:
            return 0

    def do_LaMM_plot(self, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, newfig=True, 
                   fsiz=(8,6), ax='None', myyrange=[10.,300.], myxrange=[0.03,4.], dolabel=True):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            f = plt.figure(figsize=fsiz)

        myxra = np.copy(self.mytable['Lstar']**0.25)
        for i in range(len(myxra)):
            if self.mytable['a'][i]!='...':
                myxra[i] = myxra[i]/self.mytable['Mstar'][i]**0.25/(float(self.mytable['a'][i])*self.mytable['Dist'][i])**0.5
            else:
                myxra[i] = 1.e5

        ngra = np.where(myxra[self.nval] < 1.e5)
        self.Tlt_ruvg = np.median((self.mytable[self.nval]['Tmm_50']/myxra[self.nval])[ngra])

        # Plot for Rout
        plt.errorbar(self.mytable[self.nval]['Mstar'],self.mytable[self.nval]['Tmm_50']/myxra[self.nval],
             yerr=[self.mytable[self.nval]['Tmm_em']/myxra[self.nval],self.mytable[self.nval]['Tmm_ep']/myxra[self.nval]], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        plt.errorbar(self.mytable[self.ninval]['Mstar'],self.mytable[self.ninval]['Tmm_50']/myxra[self.ninval],
             yerr=[self.mytable[self.ninval]['Tmm_em']/myxra[self.ninval],self.mytable[self.ninval]['Tmm_ep']/myxra[self.ninval]], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        plt.plot([myxrange[0],myxrange[1]],[self.Tlt_ruvg,self.Tlt_ruvg],linestyle='dotted',color='k')
        mylab = "Median = %5.2f" % (self.Tlt_ruvg)
        xl = myxrange[0]+(myxrange[1]-myxrange[0])*0.005
        yl = myyrange[0]+(myyrange[1]-myyrange[0])*0.01
        if dolabel:
            plt.text(xl,yl,mylab)
        # Y-axis
        plt.xscale('log')
        #ax[0].set_xlabel(r'(L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$')
        plt.xlabel(r'M$_\star$/M$_\odot$')
        plt.xlim(myxrange[0],myxrange[1])
        # Y-axis
        plt.ylabel(r'(T$_{mm}$/K)/(((L$_\star$/L$_\odot$)/(M$_\star$/M$_\odot$))$^{0.25}/$(R$_{uvg}$/AU)$^{0.5}$)')
        plt.yscale('log')
        plt.ylim(myyrange[0],myyrange[1])

        if newfig:
            return f 
        else:
            return 0

    def do_aM_plot(self, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, newfig=True, 
                   fsiz=(8,6), ax='None', myyrange=[10.,300.], myxrange=[0.03,4.], dolabel=True):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            f = plt.figure(figsize=fsiz)

        myxra = np.copy(self.mytable['Lstar']**0.25)
        for i in range(len(myxra)):
            if self.mytable['a'][i]!='...':
                myxra[i] = myxra[i]/(float(self.mytable['a'][i])*self.mytable['Dist'][i])**0.5
            else:
                myxra[i] = 1.e5

        ngra = np.where(myxra[self.nval] < 1.e5)
        self.Tlt_ruvg = np.median((self.mytable[self.nval]['Tmm_50']/myxra[self.nval])[ngra])

        # Plot for Rout
        plt.errorbar(self.mytable[self.nval]['Mstar'],self.mytable[self.nval]['Tmm_50']/myxra[self.nval],
             yerr=[self.mytable[self.nval]['Tmm_em']/myxra[self.nval],self.mytable[self.nval]['Tmm_ep']/myxra[self.nval]], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        plt.errorbar(self.mytable[self.ninval]['Mstar'],self.mytable[self.ninval]['Tmm_50']/myxra[self.ninval],
             yerr=[self.mytable[self.ninval]['Tmm_em']/myxra[self.ninval],self.mytable[self.ninval]['Tmm_ep']/myxra[self.ninval]], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        plt.plot([myxrange[0],myxrange[1]],[self.Tlt_ruvg,self.Tlt_ruvg],linestyle='dotted',color='k')
        mylab = "Median = %5.2f" % (self.Tlt_ruvg)
        xl = myxrange[0]+(myxrange[1]-myxrange[0])*0.005
        yl = myyrange[0]+(myyrange[1]-myyrange[0])*0.01
        if dolabel:
            plt.text(xl,yl,mylab)
        # Y-axis
        plt.xscale('log')
        #ax[0].set_xlabel(r'(L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$')
        plt.xlabel(r'M$_\star$/M$_\odot$')
        plt.xlim(myxrange[0],myxrange[1])
        # Y-axis
        plt.ylabel(r'(T$_{mm}$/K)/((L$_\star$/L$_\odot$)$^{0.25}/$(R$_{uvg}$/AU)$^{0.5}$)')
        plt.yscale('log')
        plt.ylim(myyrange[0],myyrange[1])

        if newfig:
            return f 
        else:
            return 0

    def do_aF_plot(self, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, newfig=True, 
                   fsiz=(8,6), ax='None', myyrange=[10.,300.], myxrange=[0.1,1000.], dolabel=True):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            f = plt.figure(figsize=fsiz)

        myxra = np.copy(self.mytable['Lstar']**0.25)
        for i in range(len(myxra)):
            if self.mytable['a'][i]!='...':
                myxra[i] = myxra[i]/(float(self.mytable['a'][i])*self.mytable['Dist'][i])**0.5
            else:
                myxra[i] = 1.e5

        ngra = np.where(myxra[self.nval] < 1.e5)
        self.Tlt_ruvg = np.median((self.mytable[self.nval]['Tmm_50']/myxra[self.nval])[ngra])

        # Plot for Rout
        plt.errorbar(self.mytable[self.nval]['F_cont'],self.mytable[self.nval]['Tmm_50']/myxra[self.nval],
             yerr=[self.mytable[self.nval]['Tmm_em']/myxra[self.nval],self.mytable[self.nval]['Tmm_ep']/myxra[self.nval]], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        plt.errorbar(self.mytable[self.ninval]['F_cont'],self.mytable[self.ninval]['Tmm_50']/myxra[self.ninval],
             yerr=[self.mytable[self.ninval]['Tmm_em']/myxra[self.ninval],self.mytable[self.ninval]['Tmm_ep']/myxra[self.ninval]], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        plt.plot([myxrange[0],myxrange[1]],[self.Tlt_ruvg,self.Tlt_ruvg],linestyle='dotted',color='k')
        mylab = "Median = %5.2f" % (self.Tlt_ruvg)
        xl = myxrange[0]+(myxrange[1]-myxrange[0])*0.005
        yl = myyrange[0]+(myyrange[1]-myyrange[0])*0.01
        if dolabel:
            plt.text(xl,yl,mylab)
        # Y-axis
        plt.xscale('log')
        #ax[0].set_xlabel(r'(L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$')
        plt.xlabel(r'F$_{890\mu m}$')
        plt.xlim(myxrange[0],myxrange[1])
        # Y-axis
        plt.ylabel(r'(T$_{mm}$/K)/((L$_\star$/L$_\odot$)$^{0.25}/$(R$_{uvg}$/AU)$^{0.5}$)')
        plt.yscale('log')
        plt.ylim(myyrange[0],myyrange[1])

        if newfig:
            return f 
        else:
            return 0


    def do_RM_plot(self, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, newfig=True, 
                   fsiz=(15,5), ax='None', myyrange=[2.,1000.], myxrange=[0.03,4.]):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            f, ax = plt.subplots(1, 2, figsize=fsiz)

        

        # Plot for Rout
        ax[0].errorbar(self.mytable[self.nval]['Mstar'],self.mytable[self.nval]['R_out_50']/self.mytable[self.nval]['Mstar']**0.5,
             yerr=[self.mytable[self.nval]['R_out_em'],self.mytable[self.nval]['R_out_ep']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[0].errorbar(self.mytable[self.ninval]['Mstar'],self.mytable[self.ninval]['R_out_50']/self.mytable[self.ninval]['Mstar']**0.5,
             yerr=[self.mytable[self.ninval]['R_out_em'],self.mytable[self.ninval]['R_out_ep']], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        ax[0].set_xscale('log')
        #ax[0].set_xlabel(r'(L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$')
        ax[0].set_xlabel(r'M$_\star$/M$_\odot$')
        ax[0].set_xlim(myxrange[0],myxrange[1])
        # Y-axis
        ax[0].set_ylabel(r'(R$_{out}$/AU)/(M$_\star$/M$_\odot$)$^{0.5}$')
        ax[0].set_yscale('log')
        ax[0].set_ylim(myyrange[0],myyrange[1])

        # Plot for Rc
        ax[1].errorbar(self.mytable[self.nval]['Mstar'],self.mytable[self.nval]['Rc_50']/self.mytable[self.nval]['Mstar']**0.5,
             yerr=[self.mytable[self.nval]['Rc_em'],self.mytable[self.nval]['Rc_ep']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[1].errorbar(self.mytable[self.ninval]['Mstar'],self.mytable[self.ninval]['Rc_50']/self.mytable[self.ninval]['Mstar']**0.5,
             yerr=[self.mytable[self.ninval]['Rc_em'],self.mytable[self.ninval]['Rc_ep']], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        ax[1].set_xscale('log')
        ax[1].set_xlabel(r'M$_\star$/M$_\odot$')
        ax[1].set_xlim(myxrange[0],myxrange[1])
        # Y-axis
        ax[1].set_ylabel(r'(R$_c$/AU)/(M$_\star$/M$_\odot$)$^{0.5}$')
        ax[1].set_yscale('log')
        ax[1].set_ylim(myyrange[0],myyrange[1])

        if newfig:
            return f, ax 
        else:
            return 0

    def do_RR_plot(self, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, newfig=True, 
                   fsiz=(15,5), ax='None', myyrange=[2.,1000.], myxrange1=[0.03,4.], myxrange2=[0.03,4.]):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            f, ax = plt.subplots(1, 2, figsize=fsiz)

        myxra = np.zeros(len(self.mytable['a']))
        for i in range(len(myxra)):
            if self.mytable['a'][i]!='...':
                myxra[i] = float(self.mytable['a'][i])*self.mytable['Dist'][i]
            else:
                myxra[i] = 1.e5

        ngra = np.where(myxra[self.nval] < 1.e5)

        # Plot Ruvg vs Rout
        ax[0].errorbar(self.mytable[ngra]['R_out_50'],myxra[ngra],
             yerr=[self.mytable[ngra]['e_a']*self.mytable['Dist'][i],self.mytable[ngra]['e_a']*self.mytable['Dist'][i]], 
             xerr=[self.mytable[ngra]['R_out_16'],self.mytable[ngra]['R_out_84']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        ax[0].set_xscale('log')
        #ax[0].set_xlabel(r'(L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$')
        ax[0].set_xlabel(r'R$_{out}$/AU')
        ax[0].set_xlim(myxrange1[0],myxrange1[1])
        # Y-axis
        ax[0].set_ylabel(r'(R$_{uvg}$/AU)')
        ax[0].set_yscale('log')
        ax[0].set_ylim(myyrange[0],myyrange[1])

        # Plot for Rc
        # Plot Ruvg vs Rout
        ax[1].errorbar(self.mytable[ngra]['Rc_50'],myxra[ngra],
             yerr=[self.mytable[ngra]['e_a']*self.mytable['Dist'][i],self.mytable[ngra]['e_a']*self.mytable['Dist'][i]], 
             xerr=[self.mytable[ngra]['Rc_16'],self.mytable[ngra]['Rc_84']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        ax[1].set_xscale('log')
        #ax[0].set_xlabel(r'(L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$')
        ax[1].set_xlabel(r'R$_{c}$/AU')
        ax[1].set_xlim(myxrange2[0],myxrange2[1])
        # Y-axis
        ax[1].set_ylabel(r'(R$_{uvg}$/AU)')
        ax[1].set_yscale('log')
        ax[1].set_ylim(myyrange[0],myyrange[1])

        if newfig:
            return f, ax 
        else:
            return 0

    def do_LR_plot(self, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, newfig=True, 
                   fsiz=(15,5), ax='None', myyrange=[12.,1000.], myxrange=[0.03,4.]):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            f, ax = plt.subplots(1, 2, figsize=fsiz)

        myxrout = self.mytable['Lstar']**0.25/self.mytable['R_out_50']**0.5
        myxrc = self.mytable['Lstar']**0.25/self.mytable['Rc_50']**0.5

        # Plot for Rout
        ax[0].errorbar(self.mytable[self.nval]['Mstar'],self.mytable[self.nval]['Tmm_50']/myxrout[self.nval],
             yerr=[self.mytable[self.nval]['Tmm_em']/myxrout[self.nval],self.mytable[self.nval]['Tmm_ep']/myxrout[self.nval]], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[0].errorbar(self.mytable[self.ninval]['Mstar'],self.mytable[self.ninval]['Tmm_50']/myxrout[self.ninval],
             yerr=[self.mytable[self.ninval]['Tmm_em']/myxrout[self.ninval],self.mytable[self.ninval]['Tmm_ep']/myxrout[self.ninval]], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        ax[0].set_xscale('log')
        #ax[0].set_xlabel(r'(L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$')
        ax[0].set_xlabel(r'M$_\star$/M$_\odot$')
        ax[0].set_xlim(myxrange[0],myxrange[1])
        # Y-axis
        ax[0].set_ylabel(r'(T$_{mm}$/K)/((L$_\star$/L$_\odot$)$^{0.25}/$(R$_{out}$/AU)$^{0.5}$)')
        ax[0].set_yscale('log')
        ax[0].set_ylim(myyrange[0],myyrange[1])

        # Plot for Rc
        ax[1].errorbar(self.mytable[self.nval]['Mstar'],self.mytable[self.nval]['Tmm_50']/myxrc[self.nval],
             yerr=[self.mytable[self.nval]['Tmm_em']/myxrc[self.nval],self.mytable[self.nval]['Tmm_ep']/myxrc[self.nval]], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[1].errorbar(self.mytable[self.ninval]['Mstar'],self.mytable[self.ninval]['Tmm_50']/myxrc[self.ninval],
             yerr=[self.mytable[self.ninval]['Tmm_em']/myxrc[self.ninval],self.mytable[self.ninval]['Tmm_ep']/myxrc[self.ninval]], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        ax[1].set_xscale('log')
        ax[1].set_xlabel(r'M$_\star$/M$_\odot$')
        ax[1].set_xlim(myxrange[0],myxrange[1])
        # Y-axis
        ax[1].set_ylabel(r'(T$_{mm}$/K)/((L$_\star$/L$_\odot$)$^{0.25}/$(R$_c$/AU)$^{0.5}$)')
        ax[1].set_yscale('log')
        ax[1].set_ylim(myyrange[0],myyrange[1])

        if newfig:
            return f, ax 
        else:
            return 0
        
    def do_6_par_plot(self, ax='None', newfig=True, plot_tmm_recipes=False, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, 
                      mytrange=[6.,200.], mygrange=[-1.8,2.], mylrange=[0.006,110.], mymsrange=[0.05,4.],
                      mymdrange=[1.,1000], myroutrange=[2.,600.], myrcrange=[1.,90.]):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            nx=2
            ny=3
            fig, ax = plt.subplots(ny, nx, sharex=False, sharey=False, squeeze=True, figsize=(8*nx,8*ny))

        # Plot Tmm vs Lstar
        # plot temperature recipes
        if plot_tmm_recipes:
            lls_rec = np.linspace(-3,4,100)
            ls_rec = 10**lls_rec
            myta = 25.*(ls_rec)**0.25
            mytvp = 22.*(ls_rec)**0.16 
            ax[0][0].plot([-3,5],[20.,20.],color='red',linestyle='dotted')
            ax[0][0].plot(ls_rec,myta,color='red',linestyle='solid')
            ax[0][0].plot(ls_rec,mytvp,color='red',linestyle='dashed')
        # plot data
        ax[0][0].errorbar((self.mytable[self.nval]['Lstar']),self.mytable[self.nval]['Tmm_50'],
             yerr=[self.mytable[self.nval]['Tmm_em'],self.mytable[self.nval]['Tmm_ep']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[0][0].errorbar((self.mytable[self.ninval]['Lstar']),self.mytable[self.ninval]['Tmm_50'],
             yerr=[self.mytable[self.ninval]['Tmm_em'],self.mytable[self.ninval]['Tmm_ep']], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        ax[0][0].set_xscale('log')
        ax[0][0].set_xlabel(r'L$_\star$/L$_\odot$')
        ax[0][0].set_xlim(mylrange[0],mylrange[1])
        # Y-axis
        ax[0][0].set_ylabel(r'T$_{mm}$ (K)')
        ax[0][0].set_yscale('log')
        ax[0][0].set_ylim(mytrange[0],mytrange[1])

        # Plot Tmm vs Mstar
        ax[0][1].errorbar((self.mytable[self.nval]['Mstar']),self.mytable[self.nval]['Tmm_50'],
             yerr=[self.mytable[self.nval]['Tmm_em'],self.mytable[self.nval]['Tmm_ep']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[0][1].errorbar((self.mytable[self.ninval]['Mstar']),self.mytable[self.ninval]['Tmm_50'],
             yerr=[self.mytable[self.ninval]['Tmm_em'],self.mytable[self.ninval]['Tmm_ep']], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # Y-axis
        ax[0][1].set_xscale('log')
        ax[0][1].set_xlabel(r'M$_\star$/M$_\odot$')
        ax[0][1].set_xlim(mymsrange[0],mymsrange[1])
        # Y-axis
        ax[0][1].set_ylabel(r'T$_{mm}$ (K)')
        ax[0][1].set_yscale('log')
        ax[0][1].set_ylim(mytrange[0],mytrange[1])

        # Plot Tmm vs Mdust
        ax[2][0].errorbar((self.mytable[self.nval]['M_dust_50']),self.mytable[self.nval]['Tmm_50'],
             yerr=[self.mytable[self.nval]['Tmm_em'],self.mytable[self.nval]['Tmm_ep']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[2][0].errorbar((self.mytable[self.ninval]['M_dust_50']),self.mytable[self.ninval]['Tmm_50'],
             yerr=[self.mytable[self.ninval]['Tmm_em'],self.mytable[self.ninval]['Tmm_ep']], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # X-axis
        ax[2][0].set_xscale('log')
        ax[2][0].set_xlabel(r'M$_d$/M$_\oplus$')
        ax[2][0].set_xlim(mymdrange[0],mymdrange[1])
        # Y-axis
        ax[2][0].set_ylabel(r'T$_{mm}$ (K)')
        ax[2][0].set_yscale('log')
        ax[2][0].set_ylim(mytrange[0],mytrange[1])

        # Plot Tmm vs Rout
        ax[2][1].errorbar((self.mytable[self.nval]['R_out_50']),self.mytable[self.nval]['Tmm_50'],
             yerr=[self.mytable[self.nval]['Tmm_em'],self.mytable[self.nval]['Tmm_ep']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[2][1].errorbar((self.mytable[self.ninval]['R_out_50']),self.mytable[self.ninval]['Tmm_50'],
             yerr=[self.mytable[self.ninval]['Tmm_em'],self.mytable[self.ninval]['Tmm_ep']], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # X-axis
        #ax[2][1].set_xscale('log')
        ax[2][1].set_xlabel(r'R$_{out}$ (AU)')
        ax[2][1].set_xlim(myroutrange[0],myroutrange[1])
        # Y-axis
        ax[2][1].set_ylabel(r'T$_{mm}$ (K)')
        ax[2][1].set_yscale('log')
        ax[2][1].set_ylim(mytrange[0],mytrange[1])

        # Plot Tmm vs Rc
        ax[1][0].errorbar((self.mytable[self.nval]['Rc_50']),self.mytable[self.nval]['Tmm_50'],
             yerr=[self.mytable[self.nval]['Tmm_em'],self.mytable[self.nval]['Tmm_ep']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[1][0].errorbar((self.mytable[self.ninval]['Rc_50']),self.mytable[self.ninval]['Tmm_50'],
             yerr=[self.mytable[self.ninval]['Tmm_em'],self.mytable[self.ninval]['Tmm_ep']], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # X-axis
        #ax[1][0].set_xscale('log')
        ax[1][0].set_xlabel(r'R$c$ (AU)')
        ax[1][0].set_xlim(myrcrange[0],myrcrange[1])
        # Y-axis
        ax[1][0].set_ylabel(r'T$_{mm}$ (K)')
        ax[1][0].set_yscale('log')
        ax[1][0].set_ylim(mytrange[0],mytrange[1])

        # Plot Tmm vs gamma
        ax[1][1].errorbar((self.mytable[self.nval]['gamma_50']),self.mytable[self.nval]['Tmm_50'],
             yerr=[self.mytable[self.nval]['Tmm_em'],self.mytable[self.nval]['Tmm_ep']], 
             fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        ax[1][1].errorbar((self.mytable[self.ninval]['gamma_50']),self.mytable[self.ninval]['Tmm_50'],
             yerr=[self.mytable[self.ninval]['Tmm_em'],self.mytable[self.ninval]['Tmm_ep']], 
             mfc='none', fmt=mysymbol,color=mycolor, markersize=marksiz, elinewidth=elsiz)
        # X-axis
        # ax[1][1].set_xscale('log')
        ax[1][1].set_xlabel(r'$\gamma$')
        ax[1][1].set_xlim(mygrange[0],mygrange[1])
        # Y-axis
        ax[1][1].set_ylabel(r'T$_{mm}$ (K)')
        ax[1][1].set_yscale('log')
        ax[1][1].set_ylim(mytrange[0],mytrange[1])

        #
        if newfig:
            return fig, ax
        else:
            return 0
