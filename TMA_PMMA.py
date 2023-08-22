import base64
import matplotlib.pyplot as plt
from IPython.display import display, HTML

import numpy as np
from scipy.integrate import odeint

# define TMA_PMMA model
class TMA_PMMA(object):
    """
    C + B -> S
    """

    def __init__(self, params):
        # obtain parameters for pde model
        self.l = params['l'] * 1e-4 # convert µm to cm
        self.df = params['D0']
        self.sc = params['C0surface']
        self.pc = params['C0polymer']
        self.hd = params['Kprime']
        self.k = params['k']
        self.mm = params['mm']

    def fpde(self, state, t):
        # obtain state information
        C, S, B = np.hsplit(state, 3)
        # compute boundary condition
        bdC = self.sc if (t <= self.st) else max(0, self.sc * (1-(t-self.st)/60))
        # compute time derivative
        fluxC = np.diff(np.concatenate(([bdC],C,[bdC]))) / np.concatenate(([self.dx/2],np.repeat(self.dx,2*self.xdim-1),[self.dx/2]))
        dCdt = self.df * np.exp(-self.hd * S) * np.diff(fluxC)/self.dx - self.k * C * B
        dSdt = self.k * C * B
        dBdt = -self.k * C * B
        return np.concatenate((dCdt,dSdt,dBdt))

    def init_states(self, xdim = 1000):
        self.xdim = xdim # level of discretization
        self.dx = self.l / self.xdim
        C = np.zeros(2 * self.xdim)
        S = np.zeros(2 * self.xdim)
        B = np.repeat(self.pc, 2 * self.xdim)
        return np.concatenate((C,S,B))

    def init_times(self, timepoints):
        # timepoints = (sorption_time (st), desorption_time (dt))
        self.st, self.dt = timepoints
        sorption_times = np.linspace(0, min(self.st,5000), 2001) # 2.5 seconds interval for first 5000 seconds
        if (self.st > 5025):
            steps = (self.st-5000-1)//25 # 25 seconds interval after 5000 seconds
            sorption_times = np.concatenate((sorption_times, np.linspace(5025,5025+(steps-1)*25,steps)))
        desorption_times = np.linspace(0, min(self.dt,5000), 2001) # 2.5 seconds interval for first 5000 seconds
        if (self.dt > 5000):
            steps = (self.dt-5000-1)//25 # 25 seconds interval after 5000 seconds
            desorption_times = np.concatenate((desorption_times, np.linspace(5025,5025+(steps-1)*25,steps), np.array([self.dt])))
        return np.concatenate((sorption_times, self.st+desorption_times))

    def solve(self, timepoints, xdim = 1000, rtol = 1.49012e-8, atol = 1.49012e-8):
        self.times = self.init_times(timepoints)
        s0 = self.init_states(xdim)
        self.states = odeint(self.fpde, s0, self.times, rtol=rtol, atol=atol)

    def mass_uptake_over_time(self, return_value=False):
        C, S, B = [np.mean(v,1) * self.l * self.mm * 1e9 for v in np.hsplit(self.states,3)]
        normalizer = self.pc * self.l * self.mm * 1e9
        # csv output file
        csv = 'times,Cfree,Cprod,Cpoly,normalized_Cfree,normalized_Cprod,normalized_Cpoly\n' +\
            '\n'.join(['%.18e,%.18e,%.18e,%.18e,%.18e,%.18e,%.18e'
                        %(self.times[i],C[i],S[i],B[i],C[i]/normalizer,S[i]/normalizer,B[i]/normalizer)
                        for i in range(self.times.size)])
        html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(
            payload=base64.b64encode(csv.encode()).decode(),
            title="Download CSV File",
            filename="mass_uptake.csv"
        )
        display(HTML(html))
        # visualization
        fig = plt.figure(figsize=(8,8), facecolor='w')
        ax1 = fig.add_subplot(211, facecolor='#dddddd', axisbelow=True)
        ax1.plot(self.times, (C+S)/normalizer, lw=1, color='r', label='Cfree + Cprod')
        ax1.plot(self.times, C/normalizer, lw=1, color='g', label='Cfree')
        ax1.plot(self.times, S/normalizer, lw=1, color='b', label='Cprod')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Normalized Molar Sorption of Precursor')
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.set_ylabel('Mass/Area (ng/cm^2)')
        mn, mx = ax1.get_ylim()
        ax2.set_ylim(mn*normalizer, mx*normalizer)
        plt.title('Temporal Mass Uptake of Precursor in Polymer (Linear Time)')
        ax1 = fig.add_subplot(212, facecolor='#dddddd', axisbelow=True)
        ax1.plot(np.sqrt(self.times), (C+S)/normalizer, lw=1, color='r', label='Cfree + Cprod')
        ax1.plot(np.sqrt(self.times), C/normalizer, lw=1, color='g', label='Cfree')
        ax1.plot(np.sqrt(self.times), S/normalizer, lw=1, color='b', label='Cprod')
        ax1.set_xlabel('Root Time (s^0.5)')
        ax1.set_ylabel('Normalized Molar Sorption of Precursor')
        ax1.legend()
        ax2 = ax1.twinx()
        ax2.set_ylabel('Mass/Area (ng/cm^2)')
        mn, mx = ax1.get_ylim()
        ax2.set_ylim(mn*normalizer, mx*normalizer)
        plt.title('Temporal Mass Uptake of Precursor in Polymer (Root Time)')
        fig.tight_layout()
        plt.show()
        # return value
        if (return_value):
            return (C,S,B)

    def depth_profile(self, component, process, timepoints=[], return_value=False):
        # find idx for timepoints
        if (not (isinstance(timepoints, list) or isinstance(timepoints, np.ndarray))):
            raise ValueError('timepoints must be a list or numpy array object.')
        process_map = {'Sorption':self.st, 'Desorption':self.dt}
        timepoints = self.create_timepoints(0,process_map[process],11) if (len(timepoints)==0) else timepoints
        tps = np.array(timepoints) if (process=='Sorption') else np.array(timepoints) + self.st
        timeidx = np.argmin(np.abs(tps[:,None] - self.times), axis=1)
        # find the component to visualize
        C, S, B = np.hsplit(self.states[timeidx,:]* self.l * self.mm * 1e9, 3)
        if (component == 'Cfree'):
            yval = np.hsplit(self.states[timeidx,:], 3)[0]
        elif (component == 'Cpoly'):
            yval = np.hsplit(self.states[timeidx,:], 3)[2]
        elif (component == 'Cprod'):
            yval = np.hsplit(self.states[timeidx,:], 3)[1]
        elif (component == 'Cfree+Cprod'):
            yval = np.hsplit(self.states[timeidx,:], 3)[0] + np.hsplit(self.states[timeidx,:], 3)[1]
        else:
            raise ValueError(
                "component can only be 'Cfree', 'Cpoly', 'Cprod', or 'Cfree+Cprod'," 
                f"but '{component}' is provided."
            )
        yval = yval[:,:self.xdim] * self.l * self.mm * 1e9
        yval_normalizer = self.pc * self.l * self.mm * 1e9
        normalized_yval = yval / yval_normalizer
        # space dicretization value
        xval = (np.arange(self.xdim)+0.5) * self.dx
        xval = xval * 1e4 # convert from cm to µm
        # csv output file
        # Normalized Molar Sorption of Precursor
        csv_header = 'thickness(µm),' + ','.join([str(t) for t in timepoints]) + '\n'
        csv_text = '\n'.join(['%.18e,' % xval[j] + ','.join('%18e' % normalized_yval[i,j] for i in range(len(timepoints))) for j in range(self.xdim)])
        csv = csv_header + csv_text
        html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(
            payload=base64.b64encode(csv.encode()).decode(),
            title=f"Download CSV for {component}-{process}: Normalized Molar Sorption of Precursor",
            filename=f"{component}_{process}_normalized_mass_uptake.csv"
        )
        display(HTML(html))
        # Mass/Area (ng/cm^2)
        csv_header = 'thickness(µm),' + ','.join([str(t) for t in timepoints]) + '\n'
        csv_text = '\n'.join(['%.18e,' % xval[j] + ','.join('%18e' % yval[i,j] for i in range(len(timepoints))) for j in range(self.xdim)])
        csv = csv_header + csv_text
        html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
        html = html.format(
            payload=base64.b64encode(csv.encode()).decode(),
            title=f"Download CSV for {component}-{process}: Mass/Area (ng/cm^2)",
            filename=f"{component}_{process}_absolute_mass_uptake.csv"
        )
        display(HTML(html))
        # visualization
        fig = plt.figure(figsize=(8,6), facecolor='w')
        cmap=plt.get_cmap("jet")
        ax1 = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
        for i in range(len(timepoints)):
            color = cmap(i/len(timepoints))
            ax1.plot(xval, normalized_yval[i,:], color=color, label=int(timepoints[i]))
        ax1.set_xlabel('Polymer Film Thickness (µm)')
        ax1.set_xlim(0, self.l * 1e4)
        ax1.set_ylabel('Normalized Molar Sorption of Precursor')
        ax1.set_ylim(-0.05 * max(0.10,np.max(normalized_yval)), 1.05 * max(0.10,np.max(normalized_yval)))
        ax1.legend(title='Time (s)')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Mass/Area (ng/cm^2)')
        mn, mx = ax1.get_ylim()
        ax2.set_ylim(mn*yval_normalizer, mx*yval_normalizer)
        plt.title(f"{component} - {process}")
        plt.show()
        if (return_value):
            return yval

    def create_timepoints(self, start_timepoint, stop_timepoint, number_of_points):
        return np.linspace(start_timepoint, stop_timepoint, number_of_points)