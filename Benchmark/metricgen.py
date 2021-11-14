import numpy as np

from skimage.measure import block_reduce

"""
 Metrics

This defines a bunch of metrics which will take (target, prediction), and returns the computed metrics. 
Inputs of shape (time,parameter). Will compute metric along axis 0. 

"""
def TSEvents(dbn,dbe):
  """
    NOTE: dbn and dbe are at 1-min cadence. 
    Time runs along axis=0, and stations along other axis.
    Procedure:
    1. Construct db_dt = \sqrt{(dbn/dt)**2+(dbe/dt)**2}.
    2. Choose a threshold value in nT/s. Let's say 0.3 nT/s.
    3. Chop the time series into windows of 20 minutes. 
    4. In each window, if atleast one point has db_dt>threshold, call it an event. 
  """
  db_dt = np.sqrt(np.square(np.gradient(dbn,60,axis=0))+np.square(np.gradient(dbe,60,axis=0)))
  window = 20 # IN MINUTES
  thresholds = [0.3,0.7,1.1,1.5]
  w_size=np.ones(db_dt.ndim)
  w_size[0] = 20
  w_size = tuple(w_size.astype(int))
  print(w_size)
  final = {}
  for thresh in thresholds:
    tmpar = np.copy(db_dt)
    #Get sign of db_dt-thresh, and make all elements <0 to be 0.
    tmpar = np.sign(tmpar-thresh)
    #All negative values are 0.0
    tmpar = (tmpar+np.abs(tmpar))/2.0
    # Consider windows of 20 mins. Add the sign values across the whole window.
    reduced = block_reduce(tmpar,w_size,func=np.nansum)
    # We only care if an event occured or not in a window, not the absolute values.
    events = np.sign(reduced)
    final[thresh] = events
  return final
def Confusion_matrix(targdbe,targdbn,preddbe,preddbn):
  events_targ = TSEvents(targdbe,targdbn)
  events_pred = TSEvents(preddbe,preddbn)
  metrics = {}
  for k in events_targ.keys():
    hits = np.nansum(events_targ[k]*events_pred[k],axis=0)
    misses = np.nansum(np.heaviside(events_targ[k]-events_pred[k],0.0),axis=0)
    falsealarm = np.nansum(np.heaviside(events_pred[k]-events_targ[k],0.0),axis=0)
    truenegative = np.nansum((1-events_targ[k])*(1-events_pred[k]),axis=0)
    metrics[k] = {'hits':hits,'misses':misses,'falsealarm':falsealarm,'truenegative':truenegative}
  return metrics
def EventMetrics(targdbe,targdbn,preddbe,preddbn):
  confuse = Confusion_matrix(targdbe,targdbn,preddbe,preddbn)
  event_metrics = {}
  for k in confuse.keys():
    tmpconf = confuse[k]
    pod = tmpconf['hits']/(tmpconf['hits']+tmpconf['misses'])
    pofd = tmpconf['falsealarm']/(tmpconf['falsealarm']+tmpconf['truenegative'])
    pc =  (tmpconf['hits']+tmpconf['truenegative'])/(tmpconf['hits']+ tmpconf['falsealarm']+tmpconf['misses']+tmpconf['truenegative'])
    hss = 2*(tmpconf['hits']*tmpconf['truenegative']-tmpconf['misses']*tmpconf['falsealarm'])
    hss = hss/((tmpconf['hits']+tmpconf['misses'])*(tmpconf['misses']+tmpconf['truenegative'])+(tmpconf['hits']+tmpconf['falsealarm'])*(tmpconf['falsealarm']+tmpconf['truenegative']))
    event_metrics[k]= {'pod':pod,'pofd':pofd,'pc':pc,'hss':hss}
  return event_metrics

def Checkdim(val):
    ndim = val.ndim
    if ndim==1:
      val = val[:,None]
      return val
    else:
      return val
def npRMSE(targ,pred):
    targ = Checkdim(targ)
    pred = Checkdim(pred)
    return np.sqrt(np.nanmean(np.square(targ-pred)))
def npMAE(targ,pred):
    targ = Checkdim(targ)
    pred = Checkdim(pred)
    return np.nanmean(np.abs(targ-pred))
def npR2(targ, pred):
    targ = Checkdim(targ)
    pred = Checkdim(pred)
    _sum_of_errors = np.nansum(np.power(pred - targ, 2), -1)
    _y_sum = np.nansum(targ, -1)
    _y_sq_sum = np.nansum(np.power(targ, 2), -1)
    _num_examples = targ.shape[1]
    return 1 - _sum_of_errors / (_y_sq_sum - (_y_sum ** 2) / _num_examples)
def Generate_metrics(targ,pred):
    metrics = {}
    metrics['mse'] = npRMSE(targ,pred)
    metrics['mae'] = npMAE(targ,pred)
    metrics['r2'] = np.nanmean(npR2(targ, pred))
    return metrics

