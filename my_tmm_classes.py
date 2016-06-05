import numpy as np
import scipy.optimize as so
import matplotlib.pyplot as plt
import astropy.io.ascii as aio
import astropy.units as u


#
# Clase che definisce la singola stella
class Disk(object):

    def __init__(self, tabrow):
        #
        self.star_id = tabrow['ID']
        self.star_name = tabrow['Name']
        self.dist = tabrow['Dist']
        self.teff = tabrow['Teff']
        self.lstar = tabrow['Lstar']
        self.llstar = np.log10(self.lstar)
        self.mstar = tabrow['Mstar']
        self.lmstar = np.log10(self.mstar)
        self.rout = tabrow['R_out_50']
        self.rout_16 = tabrow['R_out_16']
        self.rout_84 = tabrow['R_out_84']
        self.mdust = tabrow['M_dust_50']
        self.mdust_16 = tabrow['M_dust_16']
        self.mdust_84 = tabrow['M_dust_84']
        self.gamma = tabrow['gamma_50']
        self.gamma_16 = tabrow['gamma_16']
        self.gamma_84 = tabrow['gamma_84']
        self.rc = tabrow['Rc_50']
        self.rc_16 = tabrow['Rc_16']
        self.rc_84 = tabrow['Rc_84']
        self.sigma0 = tabrow['sigma0_50']
        self.sigma0_16 = tabrow['sigma0_16']
        self.sigma0_84 = tabrow['sigma0_84']
        self.inc = tabrow['inc_50']
        self.inc_16 = tabrow['inc_16']
        self.inc_84 = tabrow['inc_84']
        self.fcont = tabrow['F_cont']
        self.efcont = tabrow['E_cont']
        self.i = tabrow['i']
        self.ei = tabrow['e_i']
        self.mdust_ansdell = tabrow['M_dust']
        if tabrow['Valid'] == 1:
            self.valid = True 
        else:
            self.valid = False
        #
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
    
    def __init__(self, tabfile):
        #
        self.original_file = tabfile
        self.mytable = aio.read(tabfile)
        self.stars = []
        [self.stars.append(Disk(row)) for row in self.mytable]
        self.nstars = len(self.stars)
        self.fill_tmm_cols()
        self.nval = np.where(self.mytable['Valid'] == 1 )
        self.ninval = np.where(self.mytable['Valid'] == 0 )
                
    def __str__(self):
        rep = "Table of Disk observed and fitted properties, read from file %s containing %d objects (%d validated)" % (self.original_file,self.nstars,len(self.nval[0]))
        return rep
        
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
        self.mytable['Tmm_16'] = b * u.K
        self.mytable['Tmm_50'] = a * u.K
        self.mytable['Tmm_84'] = c * u.K
        self.mytable['Tmm_ep'] = d * u.K
        self.mytable['Tmm_em'] = e * u.K

    def write_table(self, outfile):
        self.mytable.write(outfile,format='ascii.ipac')
        
    def do_6_par_plot(self, ax='None', newfig=True, mycolor='blue', mysymbol='o', mymarksiz=18, myelsiz=3, 
                      mytrange=[6.,200.], mygrange=[-1.8,2.], mylrange=[0.006,110.], mymsrange=[0.05,4.],
                      mymdrange=[1.,1000], myroutrange=[2.,600.], myrcrange=[1.,90.]):
        marksiz=mymarksiz
        elsiz=myelsiz
        if newfig:
            nx=2
            ny=3
            fig, ax = plt.subplots(ny, nx, sharex=False, sharey=False, squeeze=True, figsize=(8*nx,8*ny))

        # Plot Tmm vs Lstar
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
        ax[2][0].set_xlabel(r'M$_d$/M$_\Earth$')
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
