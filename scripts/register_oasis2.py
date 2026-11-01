"""OASIS-2 FIRST-VISIT (MR1) RAW scans -> atlas (T88-like) space, matching the OASIS-1 *masked* volumes.
Per subject: average the mpr runs (rigid co-registration to mpr-1) -> N4 bias correction -> affine (Mattes MI) registration to the
skull-on OASIS-1 T88 template -> resample to 176x208x176 1 mm RAS grid -> apply the OASIS-1 brain mask.
Usage: python register_oasis2.py [--workers 4] [--only OAS2_0001_MR1]   (resumable; QC table in data/oasis2/registration_qc.csv)"""
import sys, time, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, nibabel as nib
ROOT = Path(__file__).resolve().parent.parent
EXTROOT = ROOT / "data" / "oasis2" / "extracted"; REG = ROOT / "data" / "oasis2" / "registered"; REG.mkdir(parents=True, exist_ok=True)

def to_sitk(arr, spacing):
    import SimpleITK as sitk
    im = sitk.GetImageFromArray(np.ascontiguousarray(arr.transpose(2, 1, 0)).astype(np.float32)); im.SetSpacing([float(s) for s in spacing]); return im
def load_raw(p):
    can = nib.as_closest_canonical(nib.load(p)); return np.squeeze(np.asarray(can.dataobj, dtype=np.float32)), can.header.get_zooms()[:3]

ROBUST = False

def process(sid, robust=False):
    global ROBUST; ROBUST = robust
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(3)
    out = REG / f"{sid}_t88_masked.nii.gz"
    if out.exists() and not robust: return sid, None
    t0 = time.time(); tpl = np.load(ROOT / "data" / "template" / "template_skull_T88.npy"); mask = np.load(ROOT / "data" / "template" / "brain_mask_T88.npy")
    fixed = to_sitk(tpl, (1, 1, 1)); from scipy import ndimage as ndi; fmask = sitk.Cast(to_sitk(ndi.binary_dilation(mask, iterations=6), (1, 1, 1)), sitk.sitkUInt8)
    runs = sorted(next(EXTROOT.glob(f"OAS2_RAW_PART*/{sid}")).joinpath("RAW").glob("mpr-*.nifti.img")); imgs = []
    for k, p in enumerate(runs):
        a, z = load_raw(p); imgs.append(to_sitk(a / max(np.percentile(a[a > 0], 99), 1e-6), z))
    ref = imgs[0]; acc = [ref]
    for im in imgs[1:]:   # rigid co-registration of the repeat scans onto mpr-1, then average (as in the OASIS pipeline)
        R = sitk.ImageRegistrationMethod(); R.SetMetricAsMeanSquares(); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(.15, 1)
        R.SetInterpolator(sitk.sitkLinear); R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-4, 100); R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4, 2]); R.SetSmoothingSigmasPerLevel([2, 1]); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        R.SetInitialTransform(sitk.CenteredTransformInitializer(ref, im, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS), inPlace=False)
        try: tr = R.Execute(ref, im); acc.append(sitk.Resample(im, ref, tr, sitk.sitkLinear, 0.0))
        except Exception: pass
    avg = sitk.Cast(sum(acc[1:], acc[0]) / len(acc), sitk.sitkFloat32)
    m = sitk.Cast(sitk.OtsuThreshold(avg, 0, 1), sitk.sitkUInt8)
    sh = [4, 4, 2]; n4 = sitk.N4BiasFieldCorrectionImageFilter(); n4.SetMaximumNumberOfIterations([30, 20, 10])
    small = sitk.Shrink(avg, sh); small_m = sitk.Shrink(m, sh)
    n4.Execute(small, small_m); log_bias = n4.GetLogBiasFieldAsImage(avg); avg = sitk.Cast(avg / sitk.Exp(sitk.Cast(log_bias, sitk.sitkFloat32)), sitk.sitkFloat32)
    def register(kind, init_mode, rigid_first, pitch=0.0):
        def mi(tx_type, init_tx, levels):
            R = sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(32); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(.12, 1)
            R.SetMetricFixedMask(fmask); R.SetInterpolator(sitk.sitkLinear)
            R.SetOptimizerAsRegularStepGradientDescent(1.5, 1e-5, 200, relaxationFactor=.6); R.SetOptimizerScalesFromPhysicalShift()
            R.SetShrinkFactorsPerLevel(levels[0]); R.SetSmoothingSigmasPerLevel(levels[1]); R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            R.SetInitialTransform(init_tx, inPlace=False); return R.Execute(fixed, avg), R
        mode = sitk.CenteredTransformInitializerFilter.MOMENTS if init_mode == "moments" else sitk.CenteredTransformInitializerFilter.GEOMETRY
        lv = ([4, 2, 1], [3, 1.5, 0.5])
        if rigid_first:
            t0_ = sitk.Euler3DTransform(); t0_ = sitk.CenteredTransformInitializer(fixed, avg, t0_, mode)
            if pitch: t0_.SetRotation(float(np.deg2rad(pitch)), 0.0, 0.0)
            rtx, _ = mi("rigid", t0_, ([8, 4, 2], [4, 2, 1]))
            e = sitk.Euler3DTransform(rtx.GetNthTransform(0)); init = sitk.AffineTransform(3); init.SetMatrix(e.GetMatrix()); init.SetTranslation(e.GetTranslation()); init.SetCenter(e.GetCenter())
        else:
            init = sitk.CenteredTransformInitializer(fixed, avg, sitk.AffineTransform(3), mode)
        tr, R = mi("affine", init, lv)
        res = sitk.Resample(avg, fixed, tr, sitk.sitkLinear, 0.0)
        arr = np.clip(sitk.GetArrayFromImage(res).transpose(2, 1, 0) * mask, 0, None).astype(np.float32)   # atlas brain mask applied
        return arr, float(np.corrcoef(arr[mask], tpl[mask])[0, 1]), float(R.GetMetricValue())
    attempts = [("affine", "moments", False)] if not ROBUST else [("a", "moments", False), ("a", "geometry", False), ("a", "moments", True), ("a", "geometry", True)] + [("a", "geometry", True, p) for p in (-40, -25, 25, 40)]
    best = None
    for att in attempts:
        r = register(*att)
        if best is None or r[1] > best[1]: best = r
        if not ROBUST and best[1] > .4: break
    arr, ncc, metric = best
    nib.save(nib.Nifti1Image(arr, np.eye(4)), out)
    return sid, dict(subject=sid, n_runs=len(runs), ncc=ncc, metric=metric, seconds=time.time() - t0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--only", nargs="+"); ap.add_argument("--robust", action="store_true"); a = ap.parse_args()
    sids = a.only if a.only else sorted(p.name for p in EXTROOT.glob("OAS2_RAW_PART*/OAS2_*_MR1") if p.is_dir())
    import pandas as pd
    qc_f = ROOT / "data" / "oasis2" / "registration_qc.csv"; rows = pd.read_csv(qc_f).to_dict("records") if qc_f.exists() else []
    with ProcessPoolExecutor(a.workers) as ex:
        for sid, r in ex.map(process, sids, [a.robust] * len(sids)):
            if r:
                rows.append(r); pd.DataFrame(rows).to_csv(qc_f, index=False); print(f"{sid} ncc={r['ncc']:.3f} {r['seconds']:.0f}s", flush=True)
    print("DONE")
