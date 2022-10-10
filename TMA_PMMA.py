import numpy as np
from scipy.integrate import odeint

# define TMA_PMMA model
class TMA_PMMA(object):
    """
    C + B -> S
    """
    
    def __init__(self, params):
        self.l = params['l'] * 2
        self.df = params['df']
        self.sc = params['sc']
        self.pc = params['pc']
        self.hd = params['hd']
        self.k = params['k']
    
    def init_state(self, xlim = 1000):
        self.xlim = xlim
        self.dx = self.l / (self.xlim + 1)
        C = np.zeros(xlim)
        S = np.zeros(xlim)
        B = np.repeat(self.pc, xlim)
        return np.concatenate((C,S,B))
    
    def fpde(self, state, t):
        # obtain state information
        C = state[:self.xlim]
        S = state[self.xlim:(2*self.xlim)]
        B = state[(2*self.xlim):(3*self.xlim)]
        # compute boundary condition
        bdC = self.sc if (t <= 62500) else max(0, self.sc * (1-(t-62500)/60))
        # compute time derivative
        fluxC = np.diff(np.concatenate(([bdC],C,[bdC]))) / np.concatenate(([self.dx/2],np.repeat(self.dx,self.xlim-1),[self.dx/2]))
        dCdt = self.df * np.exp(-self.hd * S) * np.diff(fluxC)/self.dx - self.k * C * B
        dSdt = self.k * C * B
        dBdt = -self.k * C * B
        return np.concatenate((dCdt,dSdt,dBdt))
    
    def integrate(self, s0, times, rtol = 1.49012e-8, atol = 1.49012e-8):
        states = odeint(self.fpde, s0, times, rtol=rtol, atol=atol)
        C = np.mean(states[:,:self.xlim], 1) * self.l * (72 * 1e9) / 2
        S = np.mean(states[:,self.xlim:(2*self.xlim)], 1) * self.l * (72 * 1e9) / 2
        B = np.mean(states[:,(2*self.xlim):(3*self.xlim)], 1) * self.l * (72 * 1e9) / 2
        return (C,S,B)