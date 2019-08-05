# MNIST - handwritten digits

This repository contains tests of the [AlignmentRepaPy repository](https://github.com/caiks/AlignmentRepaPy) using data from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The AlignmentRepaPy repository is a Python implementation of some of the *practicable inducers* described in the paper *The Theory and Practice of Induction by Alignment* at https://greenlake.co.uk/. 

## Documentation

There is an analysis of this dataset [here](https://greenlake.co.uk/pages/dataset_python_NIST). 

## Installation

The `NIST` executables require the `AlignmentRepa` module which is in the [AlignmentRepaPy repository](https://github.com/caiks/AlignmentRepaPy). See the AlignmentRepaPy repository for installation instructions of the Python compiler and libraries.

Then download the zip files or use git to get the NISTPy repository and the underlying AlignmentPy and AlignmentRepaPy repositories -
```
cd
git clone https://github.com/caiks/AlignmentPy.git
git clone https://github.com/caiks/AlignmentRepaPy.git
git clone https://github.com/caiks/NISTPy.git
```
Then use the Python installer tool `pip` to install the [Python Imaging Library](https://pypi.org/project/Pillow/), 
```
python3.5 -m pip install --user Pillow
```
Then download the dataset files, for example -
```
cd NIST
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
```

## Usage

The *practicable model induction* is described [here](https://greenlake.co.uk/pages/dataset_python_NIST_model#model2).

```
export PYTHONPATH=../AlignmentPy:../AlignmentRepaPy
cd NISTPy
python3

```
```py
from NISTDev import *

(uu,hrtr) = nistTrainBucketedIO(2)

digit = VarStr("digit")
vv = uvars(uu)
vvl = sset([digit])
vvk = vv - vvl

hr = hrev([i for i in range(hrsize(hrtr)) if i % 8 == 0],hrtr)

(wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed) = (2**10, 8, 2**10, 10, (10*3), 3, 2**8, 1, 15, 3, 5)

(uu1,df) = decomperIO(uu,vvk,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed)

```
This runs in Python 3.5 64-bit on a Ubuntu 16.04 Intel(R) Xeon(R) Platinum 8175M CPU @ 2.50GHz in 1507 seconds.
```py
summation(mult,seed,uu1,df,hr)
# (137828.7038275605, 68687.36371089712)

len(dfund(df))
134

len(fvars(dfff(df)))
515

open("NIST_model2.json","w").write(decompFudsPersistentsEncode(decompFudsPersistent(df)))
```


