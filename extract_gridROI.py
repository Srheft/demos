
import numpy as np
from pathlib import Path

def _overlaps(a,b):
    ax0,ax1,ay0,ay1=a; bx0,bx1,by0,by1=b
    return ax0<bx1 and ax1>bx0 and ay0<by1 and ay1>by0

def _rect(cx,cy,rw,rh,W,H):
    x0=min(max(int(round(cx-rw/2)),0),W-rw)
    y0=min(max(int(round(cy-rh/2)),0),H-rh)
    return x0,x0+rw,y0,y0+rh

def _nearest_count(n):
    if n<5: raise ValueError("TARGET_ROIS must be >= 5.")
    lo=max(5,1+4*((n-1)//4)); hi=lo if lo==n else lo+4
    return lo if n-lo < hi-n else hi

def _make_coords(shape,target,rh,rw,density=45):
    H,W=map(int,shape)
    if rh<=0 or rw<=0 or rh>H or rw>W: raise ValueError("Invalid ROI dimensions.")
    target=_nearest_count(target)
    selected=[(0,rw,0,rh),(W-rw,W,0,rh),(0,rw,H-rh,H),(W-rw,W,H-rh,H)]
    center=_rect(W/2,H/2,rw,rh,W,H)
    if any(_overlaps(center,r) for r in selected):
        raise ValueError("ROIs too large for four corners plus center.")
    selected.append(center)
    groups=(target-5)//4
    xs=np.linspace(rw/2,W/2-rw/2,density)
    ys=np.linspace(rh/2,H/2-rh/2,density)
    cand=[]
    for y in ys:
        for x in xs:
            g=[_rect(x,y,rw,rh,W,H),_rect(W-x,y,rw,rh,W,H),
               _rect(x,H-y,rw,rh,W,H),_rect(W-x,H-y,rw,rh,W,H)]
            if len(set(g))!=4: continue
            if any(_overlaps(r,s) for r in g for s in selected): continue
            if any(_overlaps(g[i],g[j]) for i in range(4) for j in range(i)): continue
            cand.append(g)
    def nc(r):
        x0,x1,y0,y1=r
        return np.array([((x0+x1)/2)/W,((y0+y1)/2)/H])
    added=0
    while added<groups and cand:
        sc=np.array([nc(r) for r in selected]); scores=[]
        for g in cand:
            gc=np.array([nc(r) for r in g])
            d=((gc[:,None,:]-sc[None,:,:])**2).sum(2)
            scores.append(np.sqrt(d.min(1)).min())
        g=cand.pop(int(np.argmax(scores)))
        if any(_overlaps(r,s) for r in g for s in selected): continue
        selected.extend(g); added+=1
        cand=[g for g in cand if not any(_overlaps(r,s) for r in g for s in selected)]
    selected=sorted(set(selected),key=lambda r:(r[2],r[0]))
    return np.asarray([(i,*r) for i,r in enumerate(selected)],dtype=np.int32)

def extract_rois(input_file,TARGET_ROIS,ROI_HEIGHT,ROI_WIDTH,show=False,input_key=None):
    input_file=Path(input_file)
    with np.load(input_file,allow_pickle=False) as z:
        if input_key is None:
            keys=[k for k in z.files if np.squeeze(z[k]).ndim==2]
            if not keys: raise ValueError("No 2-D image array found.")
            input_key=keys[0]
        frame=np.squeeze(z[input_key])
    if frame.ndim!=2: raise ValueError(f"Expected 2-D image; got {frame.shape}")

    coords=_make_coords(frame.shape,TARGET_ROIS,ROI_HEIGHT,ROI_WIDTH)
    rois=np.empty((len(coords),ROI_HEIGHT,ROI_WIDTH),dtype=frame.dtype)
    for i,(_,x0,x1,y0,y1) in enumerate(coords):
        rois[i]=frame[y0:y1,x0:x1]

    tag=f"NROIs{TARGET_ROIS}_ROIHEIGHT{ROI_HEIGHT}_ROIWIDTH{ROI_WIDTH}"
    output_file=input_file.with_name(f"{input_file.stem}_{tag}{input_file.suffix}")
    np.savez_compressed(output_file,roi_data=rois,coords=coords,
        detector_shape=np.asarray(frame.shape,dtype=np.int32),
        requested_roi_count=np.asarray(TARGET_ROIS,dtype=np.int32),
        actual_roi_count=np.asarray(len(coords),dtype=np.int32),
        roi_shape=np.asarray([ROI_HEIGHT,ROI_WIDTH],dtype=np.int32),
        source_key=np.asarray(input_key))

    old=input_file.stat().st_size; new=output_file.stat().st_size
    print(f"Input:  {old/2**20:.2f} MiB")
    print(f"Output: {new/2**20:.2f} MiB")
    print(f"Size reduction: {100*(1-new/old):.2f}% ({old/new:.2f}x smaller)")
    print(f"ROIs requested/actual: {TARGET_ROIS}/{len(coords)}")
    print(f"Saved: {output_file}")

    if show:
        import matplotlib.pyplot as plt
        vmin,vmax=np.percentile(rois,[1,99]); H,W=frame.shape
        fig,ax=plt.subplots(figsize=(15,10))
        for roi,(_,x0,x1,y0,y1) in zip(rois,coords):
            ax.imshow(roi,origin="upper",extent=[x0,x1,y1,y0],
                      interpolation="nearest",vmin=vmin,vmax=vmax,cmap="gray")
        ax.set(xlim=(0,W),ylim=(H,0),xlabel="Original detector x [pixel]",
               ylabel="Original detector y [pixel]",
               title=f"{len(coords)} ROIs; {ROI_HEIGHT}x{ROI_WIDTH} pixels")
        ax.set_aspect("equal"); plt.tight_layout(); plt.show()
    return output_file

if __name__=="__main__":
    INPUT_FILE="nsv455_top_lower_node5_2026-02-02T02_52_48.752793+00_00.npz"
    TARGET_ROIS=400
    ROI_HEIGHT=100
    ROI_WIDTH=200
    SHOW=True
    extract_rois(INPUT_FILE,TARGET_ROIS,ROI_HEIGHT,ROI_WIDTH,show=SHOW)
