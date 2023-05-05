# Hasan-- can you modify this based on the plotting code used when we ran the final version?

from re import L
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class Plotter():
    def __init__(self, xlabel, ylabel, title, y_in_log=False):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.y_in_log = y_in_log
        self.y = []
        self.x = []
        self.labels = []
        self.linecolors = []
        self.linestyle = []
        self.title = title

    def add_plot(self, x, y, label, linecolour, linestyle):
        self.y.append(y)
        self.x.append(x)
        self.labels.append(label)
        self.linecolors.append(linecolour)
        self.linestyle.append(linestyle)
        assert(len(self.y)==len(self.x) and len(self.x)==len(self.labels) and len(self.labels)==len(self.linecolors) and len(self.linecolors)==len(self.linestyle))
    
    def show_plot(self):
        matplotlib.rcParams.update({'font.size': 15})
        matplotlib.rcParams['legend.numpoints'] = 2
        fig, ax = plt.subplots(1, figsize=(8,4))
        if self.y_in_log:
            ax.set_yscale('log')
        plt.ylabel(self.ylabel)
        plt.xlabel(self.xlabel)
        matplotlib.rcParams.update({'font.size': 15})
        matplotlib.rcParams['legend.numpoints'] = 2
        for idx in range(len(self.y)):
            markersize = 13 if self.linestyle[idx] == 'dotted' else 9
            ax.plot(self.x[idx], self.y[idx], label=self.labels[idx], linestyle=self.linestyle[idx], color=self.linecolors[idx], markersize=markersize)
        matplotlib.rcParams.update({'font.size': 7})
        ax.grid(linestyle='--')
        legend = ax.legend(loc='upper left')
        plt.title(self.title)
        plt.show()

def get_accs(expdir, timesteps):
    x, pretestacc, cltestacc, totacc = [], [], [], []
    for i in range(timesteps):
        try: 
            labels1, preds1 = np.load(expdir+'/labels_'+str(i+1)+'_pretestset.npy'), np.load(expdir+'/preds_'+str(i+1)+'_pretestset.npy')
            acc = ((labels1==preds1)*1.0).mean()*100
            pretestacc.append(acc)
            labels2, preds2 = np.load(expdir+'/labels_'+str(i+1)+'_cltestset.npy'), np.load(expdir+'/preds_'+str(i+1)+'_cltestset.npy')
            acc = ((labels2==preds2)*1.0).mean()*100
            cltestacc.append(acc)
            labelsconcat, predsconcat = np.concatenate((labels1, labels2), axis=0), np.concatenate((preds1, preds2), axis=0)
            acc = ((labelsconcat==predsconcat)*1.0).mean()*100
            totacc.append(acc)
            x.append(i+1)
        except:
            continue
    pretestacc, cltestacc, totacc = np.array(pretestacc), np.array(cltestacc), np.array(totacc)
    return x, pretestacc, cltestacc, totacc

DATADIR='/Users/ameyapan/BIGILBench_Results/sampling_exps/'
linecolours = ['r','b','k','m','darkorange','g','y','c','olive','lime']

x, preacc, clacc, totacc = get_accs(expdir=DATADIR+'/CLSINC_Uniform_3000_0.1/', timesteps=20)
plter.add_plot(x=x, y=preacc, label='Uniform -- 3000 Iters Optimal -- (Im1K)', linecolour=linecolours[6], linestyle='dashed')
plter.add_plot(x=x, y=clacc, label='Uniform -- 3000 Iters Optimal -- (CL)', linecolour=linecolours[6], linestyle='solid')
plter.add_plot(x=x, y=totacc, label='Uniform -- 3000 Iters (Total)', linecolour=linecolours[6], linestyle='dotted')

plter.show_plot()