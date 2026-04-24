#https://github.com/troykn2010/py4mxrd.git. Pulled on April 8 2026

import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from scipy.ndimage import gaussian_filter1d

def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))    

def NGaussiansError(params,x,y):
    z = -y
    for i in range(0,len(params),3):
        A = params[i]
        q = params[i+1]
        s = params[i+2]
        z += gauss(x,A,q,s)
    return np.sum(z**2)

def NGaussiansClusterError(params,x,y):
    z = -y
    s = params[-1]
    for i in range(0,len(params)-1,2):
        A = params[i]
        q = params[i+1]
        z += gauss(x,A,q,s)
    return np.sum(z**2)

class MuscleLineData():

    def __init__(self,q,y,quiet = True):
        self.q = q
        self.values = y
        self.filtered_values = None
        self.background = None
        self.quiet = quiet  
        self.peaks = {}
        self.fitted_values = None


    def FitSingleGaussian(self,x,y,label = None,bounds = None, fitmethod = 'Nelder-Mead',tol=1e-8,maxiter=1e4,quiet=True):
        """
            Presume a clean gaussian that's roughly centered.
        """
        if y.max()<1e-4:
            #Error handling for vector input of zeros.
            m0 = 0
            m1 = m0
            m2 = 0
            success = False

        amplitude = y.max()
        y = y/y.max()
        a0 = np.trapz(y,x) +1e-10
        a1 = np.trapz(y*x,x) +1e-10

        p0 = [1,a1/a0,(x.max()-x.min())/6] #[0th,1st,2nd moments]
        if bounds == None:
            bounds =[(0,1) ,(x.min(),x.max()),(1e-5,(x.max()-x.min())/2+1e-10)]

        fit = minimize(NGaussiansError,p0,args = (x,y),bounds = bounds,
        method = fitmethod ,tol = tol,options={'maxiter':maxiter})
        success = fit.success
        m0 = amplitude*fit.x[0] #unpack moments
        m1 = fit.x[1]
        m2 = fit.x[2]
        if label == None:
            label = f"{1/m1:0.3f}"
        self.peaks[label] = {}
        self.peaks[label]['m2'] = m2
        self.peaks[label]['m1'] = m1
        self.peaks[label]['m0'] = m0
        self.peaks[label]['Area'] = np.sqrt(2*np.pi)*m0*m2
        self.peaks[label]['fitsuccess'] = success
        self.peaks[label]['qmin'] = bounds[1][0]
        self.peaks[label]['qmax'] = bounds[1][1]+1e-6

        if quiet==False:
            print(fit.success)
        return self.peaks[label]

    def NGaussianFit(self,listpeaks,delta= 0.5,method = 'Nelder-Mead',tol=1e-8,maxiter=1e4):     
        #Use gaussian peaks from generated from FitSingleGaussian as an initial guess for an N-Gaussian fit.
        # Best when multiple gaussians are overlapping
        a = 1 + delta
        b = 1 - delta
        p0 = []
        bnd = []

        qmin_all = self.q.max()
        qmax_all = self.q.min()
        for peak in listpeaks:
            qmin_all = min(qmin_all,peak['qmin'])
            qmax_all = max(qmax_all,peak['qmax'])
            m0 = peak['m0']
            m1 = peak['m1']
            m2 = peak['m2']
            p0 = p0 +[m0,m1,m2+1e-10] #m2=0 generates an division by zero error

            bnd= bnd + [(m0*b,m0*a),
                        (peak['qmin'],peak['qmax']), 
                        (peak['smin'],peak['smax'])
                        ]
        #Each peak has a qmin,qmax pair that defines the sandbox limits.
        #Take the largest range that contains each peak's limits
        bool = np.logical_and(self.q>qmin_all,self.q<qmax_all)
        fit = minimize(NGaussiansError,p0,args = (self.q[bool],self.filtered_values[bool]),bounds = bnd,
         method = method ,tol = tol,options={'maxiter':maxiter})
        
        #update peaks
        j = 0
        for i in range(0,len(fit.x),3):
            listpeaks[j]['m0'] = fit.x[i]
            listpeaks[j]['m1'] = fit.x[i+1]
            listpeaks[j]['m2'] = fit.x[i+2]
            listpeaks[j]['Area'] = np.sqrt(2*np.pi)*fit.x[i]*fit.x[i+2]
            listpeaks[j]['fitsuccess'] = fit.success
            j = j+1
        return listpeaks

    def FitClusterWithGaussians(self,keys,delta= 0.5,method = 'Nelder-Mead',tol=1e-8,maxiter=1e4):     
        #Use gaussian peaks from generated from FitSingleGaussian as an initial guess for an N-Gaussian fit.
        # Best when multiple gaussians are overlapping
        listpeaks = []
        for key in keys:
            listpeaks.append(self.peaks[key])

        a = 1 + delta
        b = 1 - delta
        p0 = []
        bnd = []

        qmin_all = self.q.max()
        qmax_all = self.q.min()
        for peak in listpeaks:
            qmin_all = min(qmin_all,peak['qmin'])
            qmax_all = max(qmax_all,peak['qmax'])
            m0 = peak['m0']
            m1 = peak['m1']
            m2 = peak['m2']
            p0 = p0 +[m0,m1] #m2=0 generates an division by zero error

            bnd= bnd + [(m0*b,m0*a),
                        (peak['qmin'],peak['qmax'])
                        ]
        p0 = p0 + [m2+1e-10]
        bnd = bnd + [(peak['smin'],peak['smax'])]

        #Each peak has a qmin,qmax pair that defines the sandbox limits.
        #Take the largest range that contains each peak's limits
        bool = np.logical_and(self.q>qmin_all,self.q<qmax_all)
        fit = minimize(NGaussiansClusterError,p0,args = (self.q[bool],self.filtered_values[bool]),bounds = bnd,
         method = method ,tol = tol,options={'maxiter':maxiter})

        #update peaks
        j = 0
        for i in range(0,len(fit.x)-1,2):
            listpeaks[j]['m0'] = fit.x[i]
            listpeaks[j]['m1'] = fit.x[i+1]
            listpeaks[j]['m2'] = fit.x[-1]
            listpeaks[j]['Area'] = np.sqrt(2*np.pi)*fit.x[i]*fit.x[-1]
            listpeaks[j]['fitsuccess'] = fit.success
            j = j+1

        for i,key in enumerate(keys):
            self.peaks[key]= listpeaks[i]

    def NGaussianFitKeys(self,keys,**kwargs):
        #Wrapper around NGaussianFit to take in keys as argument
        listpeaks = []
        for key in keys:
            listpeaks.append(self.peaks[key])
        newlistpeaks = self.NGaussianFit(listpeaks,**kwargs)
        for i,key in enumerate(keys):
            self.peaks[key]= newlistpeaks[i]

    def Peak_Data(self,peak):
        if peak['m2']>1e-6:
            return gauss(self.q, peak['m0'], peak['m1'], peak['m2'])
        else:
            return 0*self.q

    def BackgroundRemoval(self,interpolator):
        # self.backgroundinterpolator = deepcopy(interpolator)
        self.background = interpolator(self.q)
        self.filtered_values = self.values - self.background

    def ComputeFittedValues(self,keys):
        self.fitted_values = 0
        for key in keys:
            self.fitted_values += self.Peak_Data(self.peaks[key])

    def copy(self):
        return deepcopy(self)