import os
import subprocess
import time
import random
from pathlib import Path

def pending():
  output = subprocess.check_output('squeue -u nimamo'.split())
  output = output.decode('utf8').split('\n')
  cc = 0
  for ll in output:
    if 'PD' in ll:
      cc += 1
  return cc


confDicts = []

dataset = '4_digits_per_client'
clients_per_round = 10
secure = 1
quant = 10
clip = 0

#for dd in range(1, 11):
  #dataset = '{}_digits_per_client'.format(dd)
  #local_iters = 10
  #secure = 0
  #epsilon = 0
  #gamma = 0
  #clip = 0
  #confDicts.append({
    #'dataset': dataset,
    #'local_iters': local_iters,
    #'clients_per_round': clients_per_round,
    #'epsilon': epsilon,
    #'gamma': gamma,
    #'secure': secure,
    #'clip': clip,
    #'quant': quant
  #})
  

#for local_iters in [10, 20, 30, 40, 50, 60, 70, 90, 100]:
for local_iters in [100]:
  secure = 0
  epsilon = 0
  gamma = 0
  clip = 0
  confDicts.append({
    'num_iters': max(local_iters * 21, 10000),
    'dataset': dataset,
    'local_iters': local_iters,
    'clients_per_round': clients_per_round,
    'epsilon': epsilon,
    'gamma': gamma,
    'secure': secure,
    'clip': clip,
    'quant': quant
  })


#for clients_per_round in [1, 10, 100]:
  #local_iters = 10
  #secure = 0
  #epsilon = 0
  #gamma = 0
  #clip = 0
  #confDicts.append({
    #'dataset': dataset,
    #'local_iters': local_iters,
    #'clients_per_round': clients_per_round,
    #'epsilon': epsilon,
    #'gamma': gamma,
    #'secure': secure,
    #'clip': clip,
    #'quant': quant
  #})
  
#print()


#for local_iters in [10]:
  #for clients_per_round in [1, 10]:
    #for epsilon in [.1, .2, .5, 1.0, 1.5, 2.0]:
      #for gamma in [.2, .4, .6, .8]:
        #for clip in [.5, 1.0, 2.0]:
          #for quant in [10]:
            #confDicts.append({
              #'dataset': dataset,
              #'local_iters': local_iters,
              #'clients_per_round': clients_per_round,
              #'epsilon': epsilon,
              #'gamma': gamma,
              #'secure': secure,
              #'clip': clip,
              #'quant': quant
            #})

#for local_iters in [10, 20, 30, 40, 50, 60, 70, 90, 100]:
  #for clients_per_round in [10]:
    #for epsilon in [.2, .5]:
      #for gamma in [.2, .6]:
        #for clip in [1.0]:
          #for quant in [10]:
            #confDicts.append({
              #'num_iters': max(local_iters * 21, 2000),
              #'dataset': dataset,
              #'local_iters': local_iters,
              #'clients_per_round': clients_per_round,
              #'epsilon': epsilon,
              #'gamma': gamma,
              #'secure': secure,
              #'clip': clip,
              #'quant': quant
            #})


command = "sbatch --export={} ./job_snnfl.sh"
random.shuffle(confDicts)
try:
  os.mkdir('./slurm_lock')
except:
  pass
for ii, cc in enumerate(confDicts):
  explist = []
  explist.append("dataset='{}'".format(cc['dataset']))
  explist.append("local_iters='{}'".format(cc['local_iters']))
  explist.append("clients_per_round='{}'".format(cc['clients_per_round']))
  explist.append("epsilon='{}'".format(cc['epsilon']))
  explist.append("gamma='{}'".format(cc['gamma']))
  explist.append("secure='{}'".format(cc['secure']))
  explist.append("clip='{}'".format(cc['clip']))
  explist.append("quant='{}'".format(cc['quant']))
  if 'num_iters' in cc:
    explist.append("num_iters='{}'".format(cc['num_iters']))
  exportline = ','.join(explist)
  
  humr_fname = exportline.replace("'","").replace(',','').replace('=','')
  path_lock = os.path.join('./slurm_lock', humr_fname)
  if os.path.exists(path_lock):
    pass#continue
  Path(path_lock).touch()
  print(ii, command.format(exportline))
  print('ppending', pending())
  while True:
    if pending() < 5:
      break
    time.sleep(5)
  time.sleep(4)
  _ = subprocess.Popen(command.format(exportline).split())
print()
