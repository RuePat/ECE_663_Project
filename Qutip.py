# %%
import qutip as qtp
from matplotlib import pyplot as plt
import numpy as np
import itertools
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d
# %%
def amp_channel_ops(p):
    E0 = qtp.Qobj([[1,0],[0,np.sqrt(1-p)]])
    E1 = qtp.Qobj([[0,np.sqrt(p)],[0,0]])
    return E0,E1
def phase_channel_ops(p):
    E0 = qtp.Qobj([[1,0],[0,np.sqrt(1-p)]])
    E1 = qtp.Qobj([[0,0],[0,np.sqrt(p)]])
    return E0,E1
def depol_channel_ops(p):
    X = qtp.sigmax()*p/4
    Y = qtp.sigmay()*p/4
    Z = qtp.sigmaz()*p/4
    I = qtp.identity(2)*p/4
    return I,X,Y,Z
def sim_depol_channel(rho0,steps,p):
    rhos = [rho0]*steps
    for i in range(steps-1):
        rhos[i+1] = p*qtp.identity(rhos[i].shape[0])/2 + (1-p)*rhos[i]
    return rhos
    
def sim_channel(rho0,Es,steps):
    rhos = [rho0]*steps
    for i in range(steps-1):
        new_rho = 0*rho0 # quick zero matrix
        for E in Es:
            new_rho = new_rho + E*rhos[i]*E.dag()
        rhos[i+1] = new_rho
    return rhos

def plot_channel(rhos_sim,steps,rho_labels,cm=mpl.cm.viridis):
    row = int(np.floor(np.sqrt(len(rho_labels))))
    col = int(np.ceil(len(rho_labels)/row))
    fig,axs = plt.subplots(row,col,subplot_kw={'projection':'3d'})
    axs = axs.ravel()
    norm = mpl.colors.Normalize(vmin=0,vmax=steps)
    sm = mpl.cm.ScalarMappable(norm=norm,cmap=cm)
    steps_list = list(range(steps))
    for i,(r,l) in enumerate(zip(rhos_sim,rho_labels)):
        b = qtp.Bloch(axes=axs[i])
        b.set_label_convention('01')
        b.point_color = cm(norm(steps_list))
        b.add_states(r,kind='point')
        
        # Have final point be a star
        b.point_marker = ['.']*(len(r)-1) + ['*']
        
        b.point_size = [50]
        b.font_size = 10
        
        b.xlabel = [r"$|+\rangle$",""]
        b.ylabel = [r"$|i\rangle$",""]
        b.render()
        axs[i].set_title(rf"$\rho_0={l}$")
    fig.subplots_adjust(hspace=0.5)
    fig.colorbar(sm,ax=axs,label='Time step',orientation='horizontal')
    
    return fig

p = 0.1
ket0 = qtp.ket('0')
ket1 = qtp.ket('1')
ketp = (ket0 + ket1)/np.sqrt(2)
ketm = (ket0 - ket1)/np.sqrt(2)
ketpi = (ket0 + 1j*ket1)/np.sqrt(2)
ketmi = (ket0 - 1j*ket1)/np.sqrt(2)
ketHp = qtp.hadamard_transform().eigenstates()[1][0]
ketHm = qtp.hadamard_transform().eigenstates()[1][1]
print(ketp)
print(ketm)
print(ketpi)
print(ketmi)
print(ketHp)
print(ketHm)
rho_labels = [rf"|{s}\rangle\langle {s}|" for s in ['0','1','+','-','i','-i','H','-H']]
kets = [ket0, ket1, ketp, ketm, ketpi, ketpi,ketHp,ketHm]
rhos = list(map(qtp.ket2dm,kets))

steps = 50

amp_sim = []
for i,rho in enumerate(rhos):
    amp_sim.append(sim_channel(rho,amp_channel_ops(p),steps))
phase_sim = []
for i,rho in enumerate(rhos):
    phase_sim.append(sim_channel(rho,phase_channel_ops(p),steps))
depol_sim = []
for i,rho in enumerate(rhos):
    depol_sim.append(sim_depol_channel(rho,steps,p))


# %%
fig_amp = plot_channel(amp_sim,steps,rho_labels)
fig_amp.suptitle('Amplitude Damping Channel')
fig_phase = plot_channel(phase_sim,steps,rho_labels)
fig_phase.suptitle('Phase Damping Channel')
fig_depol = plot_channel(depol_sim,steps,rho_labels)
fig_depol.suptitle('Depolarizing Channel')
#%%
fig_amp.savefig(r'amp.jpg')
fig_phase.savefig(r'phase.jpg')
fig_depol.savefig(r'depol.jpg')