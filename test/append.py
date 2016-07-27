import numpy as np
import h5py

a = np.array([1,2,3,4,5,2,3,4,5,6]).reshape((2,5))
b = np.array([6,7,8,9,10]).reshape((1,5))
with h5py.File('tmp','w') as hf:
  hf.create_dataset('0',(0,0),compression='gzip',maxshape=(None,None))
  hf.create_dataset('1',(1,5),compression='gzip')

with h5py.File('tmp','r') as hf:
  z = np.array(hf['0'])
  print z,z.shape
  z = np.array(hf['1'])
  print z,z.shape

with h5py.File('tmp','a') as hf:
  print hf['0'].shape
  hf['0'].resize((a.shape))
  print hf['0'].shape
  hf['0'][...] = a
  hf['0'].resize((4,5))
  hf['0'][3:4,:]=b

with h5py.File('tmp','r') as hf:
  z = np.array(hf['0'])
  print z,z.shape

with h5py.File('tmp','a') as hf:
  hf['0'].resize((0,0))
  print hf['0'][...],hf['0'][...].shape
del a,b
