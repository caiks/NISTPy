from AlignmentDevRepa import *
from PIL import Image
import gzip
import struct

def aahr(uu,aa):
    return hhhr(uu,aahh(aa))

def hrlent(uu,hh,ww,vvl):
    def red(aa,vv):
        return setVarsHistoryRepasCountApproxs(vv,aa)
    z = historyRepasSize(hh)
    def ent(xx):
        t = 0
        for i in xx:
            a = i/z
            t += a * log(a)
        return - t
    return ent(red(hh,ww|vvl)) - ent(red(hh,ww))

def decomperIO(uu,vv,hr,wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed):
    return parametersSystemsHistoryRepasDecomperMaxRollByMExcludedSelfHighestFmaxIORepa(wmax,lmax,xmax,omax,bmax,mmax,umax,pmax,fmax,mult,seed,uu,vv,hr)

def qqhr(d,uu,vv,qq):
    return aahr(uu,single(llss([(v, ValInt(d-1)) for v in qq] + [(v, ValInt(0)) for v in (vv-qq)]),1))

def bmempty(sx,sy):
    return np.zeros((sx,sy,3),'uint8')

def bmfull(sx,sy):
    return np.ones((sx,sy,3),'uint8')*255

def bminsert(bm2,ox,oy,bm1):
    bm3 = np.copy(bm2)
    bm3[ox:ox+bm1.shape[0],oy:oy+bm1.shape[1],:] = bm1
    return bm3

def bmmax(bm2,ox,oy,bm1):
    bm3 = np.copy(bm2)
    bm3[ox:ox+bm1.shape[0],oy:oy+bm1.shape[1],:] = np.maximum(bm1,bm2[ox:ox+bm1.shape[0],oy:oy+bm1.shape[1],:])
    return bm3

def bmmin(bm2,ox,oy,bm1):
    bm3 = np.copy(bm2)
    bm3[ox:ox+bm1.shape[0],oy:oy+bm1.shape[1],:] = np.minimum(bm1,bm2[ox:ox+bm1.shape[0],oy:oy+bm1.shape[1],:])
    return bm3

def bmborder(b,bm):
    return bminsert(bmfull(b*2+bm.shape[0],b*2+bm.shape[1]),b,b,bm)

def bmhstack(ll):
    return np.hstack(ll)

def bmvstack(ll):
    return np.vstack(ll)

def bmwrite(file,bm):
    Image.fromarray(bm).save(file)

def hrbm(b,c,d,hr):
    z = hrsize(hr)
    (_,_,_,rr) = hr
    ar1 = np.sum(rr,axis=1)
    ar2 = ar1 * 255 // (d-1) // z
    ar3 = ar2.reshape([b,b])
    ar4 = np.transpose(np.reshape(ar3[:,:,np.newaxis]*([1]*c),[b,b*c]))
    ar5 = np.transpose(np.reshape(ar4[:,:,np.newaxis]*([1]*c),[b*c,b*c]))
    bm1 = (ar5[:,:,np.newaxis]*([1]*3)).astype(dtype='uint8')
    bm2 = bmempty(b*c,b*c)
    return bminsert(bm2,0,0,bm1)

def hrbmrow(b,c,d,hr):
    z = hrsize(hr)
    (_,_,_,rr) = hr
    ar1 = np.sum(rr,axis=1)
    ar2 = ar1 * 255 // (d-1) // z
    ar3 = ar2.reshape([1,b])
    ar4 = np.transpose(np.reshape(ar3[:,:,np.newaxis]*([1]*1),[1,b*1]))
    ar5 = np.transpose(np.reshape(ar4[:,:,np.newaxis]*([1]*c),[b*1,1*c]))
    bm1 = (ar5[:,:,np.newaxis]*([1]*3)).astype(dtype='uint8')
    bm2 = bmempty(b*1,b*1)
    return bminsert(bm2,(b-c)//2,0,bm1)

def hrbmcol(b,c,d,hr):
    z = hrsize(hr)
    (_,_,_,rr) = hr
    ar1 = np.sum(rr,axis=1)
    ar2 = ar1 * 255 // (d-1) // z
    ar3 = ar2.reshape([b,1])
    ar4 = np.transpose(np.reshape(ar3[:,:,np.newaxis]*([1]*c),[b,1*c]))
    ar5 = np.transpose(np.reshape(ar4[:,:,np.newaxis]*([1]*1),[1*c,b*1]))
    bm1 = (ar5[:,:,np.newaxis]*([1]*3)).astype(dtype='uint8')
    bm2 = bmempty(b*1,b*1)
    return bminsert(bm2,0,(b-c)//2,bm1)

def ppbm(uu,vvk,b,c,d,pp):
    kk = []
    for ll in pp:
        jj = []
        for ((_,ff),hrs) in ll:
            jj.append(bmborder(1,bmmax(hrbm(b,c,d,hrhrred(hrs,vvk)),0,0,hrbm(b,c,d,qqhr(d,uu,vvk,fund(ff))))))
        kk.append(bminsert(bmempty((b*c)+2,((b*c)+2)*max([len(ii) for ii in pp])),0,0,bmhstack(jj)))
    return bmvstack(kk)

def ppbm2(uu,vvk,b,c,d,pp):
    kk = []
    for ll in pp:
        jj = []
        for ((_,ff),hrs) in ll:
            jj.append(bmborder(1,hrbm(b,c,d,hrhrred(hrs,vvk))))
        kk.append(bminsert(bmempty((b*c)+2,((b*c)+2)*max([len(ii) for ii in pp])),0,0,bmhstack(jj)))
    return bmvstack(kk)


# nistTrainBucketedAveragedIO :: Int -> Int -> Int -> IO (System, HistoryRepa)

def nistTrainBucketedAveragedIO(d,b,q):
    def lluu(ll):
        return listsSystem([(v,sset(ww)) for (v,ww) in ll])
    uvals = systemsVarsSetValue
    f = gzip.open('train-images-idx3-ubyte.gz','rb')
    _, z, rows, cols = struct.unpack(">IIII", f.read(16))
    p = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z*rows*cols])
    c = rows // b
    ii = np.fromfunction(lambda u, x1, x2, x3: u*rows*cols + (x1*c+q+(x3//c))*cols + x2*c+q+(x3%c), (z,b,b,c*c), dtype=int)
    r = np.sum(p[ii],axis=-1).reshape([z,b*b]).astype(dtype='int32') * d // (c*c*256)
    f = gzip.open('train-labels-idx1-ubyte.gz','rb')
    _ = f.read(8)
    l = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,1])
    h = np.concatenate((np.transpose(l),np.transpose(r)))
    uu = lluu([(VarStr("digit"),[ValInt(i) for i in range(10)])] + [(VarPair((VarInt(x), VarInt(y))), [ValInt(i) for i in range(d)]) for x in range(1,b+1) for y in range(1,b+1)])
    vv = list(uvars(uu))
    mvv = sdict([(v,i) for (i,v) in enumerate(vv)])
    mm = sdict([(v,sdict([(w,i) for (i,w) in enumerate(uvals(uu,v))])) for v in vv])
    sh = tuple([len(mm[v]) for v in vv])
    hr = (vv,mvv,sh,h)
    return (uu,hr)

# nistTrainBucketedIO :: Int -> IO (System, HistoryRepa)

def nistTrainBucketedIO(d):
    def lluu(ll):
        return listsSystem([(v,sset(ww)) for (v,ww) in ll])
    uvals = systemsVarsSetValue
    f = gzip.open('train-images-idx3-ubyte.gz','rb')
    _, z, rows, cols = struct.unpack(">IIII", f.read(16))
    p = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,rows*cols])
    r = p * d // 256
    f = gzip.open('train-labels-idx1-ubyte.gz','rb')
    _ = f.read(8)
    l = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,1])
    h = np.concatenate((np.transpose(l),np.transpose(r)))
    uu = lluu([(VarStr("digit"),[ValInt(i) for i in range(10)])] + [(VarPair((VarInt(x), VarInt(y))), [ValInt(i) for i in range(d)]) for x in range(1,rows+1) for y in range(1,cols+1)])
    vv = list(uvars(uu))
    mvv = sdict([(v,i) for (i,v) in enumerate(vv)])
    mm = sdict([(v,sdict([(w,i) for (i,w) in enumerate(uvals(uu,v))])) for v in vv])
    sh = tuple([len(mm[v]) for v in vv])
    hr = (vv,mvv,sh,h)
    return (uu,hr)

# nistTrainIO :: IO (System, HistoryRepa)

def nistTrainIO():
    return nistTrainBucketedIO(256)

# nistTrainBucketedRectangleRandomIO :: Int -> Int -> Int -> Int -> IO (System, HistoryRepa)

def nistTrainBucketedRectangleRandomIO(d,bx,by,s):
    def lluu(ll):
        return listsSystem([(v,sset(ww)) for (v,ww) in ll])
    uvals = systemsVarsSetValue
    f = gzip.open('train-images-idx3-ubyte.gz','rb')
    _, z, rows, cols = struct.unpack(">IIII", f.read(16))
    p = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,rows,cols])
    r = p * d // 256
    np.random.seed(s)
    if rows-bx > 0:
        vrx = np.random.randint(0,rows-bx,z)
    else:
        vrx = np.zeros(z,dtype=np.int)
    if cols-by > 0:
        vry = np.random.randint(0,cols-by,z)
    else:
        vry = np.zeros(z,dtype=np.int)
    r = np.concatenate([r[i:i+1,vrx[i]:vrx[i]+bx,vry[i]:vry[i]+by] for i in range(z)]).reshape([z,bx*by])
    f = gzip.open('train-labels-idx1-ubyte.gz','rb')
    _ = f.read(8)
    l = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,1])
    h = np.concatenate((np.transpose(l),np.transpose(r)))
    uu = lluu([(VarStr("digit"),[ValInt(i) for i in range(10)])] + [(VarPair((VarInt(x), VarInt(y))), [ValInt(i) for i in range(d)]) for x in range(1,bx+1) for y in range(1,by+1)])
    vv = list(uvars(uu))
    mvv = sdict([(v,i) for (i,v) in enumerate(vv)])
    mm = sdict([(v,sdict([(w,i) for (i,w) in enumerate(uvals(uu,v))])) for v in vv])
    sh = tuple([len(mm[v]) for v in vv])
    hr = (vv,mvv,sh,h)
    return (uu,hr)

# nistTrainBucketedRegionRandomIO :: Int -> Int -> Int -> IO (System, HistoryRepa)

def nistTrainBucketedRegionRandomIO(d,b,s):
    return nistTrainBucketedRectangleRandomIO(d,b,b,s)

# nistTestBucketedIO :: Int -> IO (System, HistoryRepa)

def nistTestBucketedIO(d):
    def lluu(ll):
        return listsSystem([(v,sset(ww)) for (v,ww) in ll])
    uvals = systemsVarsSetValue
    f = gzip.open('t10k-images-idx3-ubyte.gz','rb')
    _, z, rows, cols = struct.unpack(">IIII", f.read(16))
    p = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,rows*cols])
    r = p * d // 256
    f = gzip.open('t10k-labels-idx1-ubyte.gz','rb')
    _ = f.read(8)
    l = np.frombuffer(f.read(),np.dtype('ubyte')).astype(dtype='int32').reshape([z,1])
    h = np.concatenate((np.transpose(l),np.transpose(r)))
    uu = lluu([(VarStr("digit"),[ValInt(i) for i in range(9)])] + [(VarPair((VarInt(x), VarInt(y))), [ValInt(i) for i in range(d)]) for x in range(1,rows+1) for y in range(1,cols+1)])
    vv = list(uvars(uu))
    mvv = sdict([(v,i) for (i,v) in enumerate(vv)])
    mm = sdict([(v,sdict([(w,i) for (i,w) in enumerate(uvals(uu,v))])) for v in vv])
    sh = tuple([len(mm[v]) for v in vv])
    hr = (vv,mvv,sh,h)
    return (uu,hr)

# nistTrainIO :: IO (System, HistoryRepa)

def nistTestIO():
    return nistTestBucketedIO(256)

def dfIO(f):
    return persistentsDecompFud_u(json.load(open(f, 'r')))

