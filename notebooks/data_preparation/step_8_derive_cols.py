import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import nibabel as nib
from pathlib import Path
import copy
from scipy import ndimage

def split_hemispheres(seg: np.array) -> np.array:
    '''Take synthseg output and return seg of L vs R (0: bg, 1: L, 2: R)'''
    L_labs = [2,3,4,5,7,8,10,11,12,13,17,18,26,28]
    R_labs = [41,42,43,44,46,47,49,50,51,52,53,54,58,60]
    N_labs = [14,15,16,24]
    
    mask = np.zeros_like(seg)
    for l in L_labs:
        mask[seg==l]=1
    for l in R_labs:
        mask[seg==l]=2
    for l in N_labs:
        mask[seg==l]=-1
    
    # fill neutral labels with nearest L or R label
    known_mask = np.isin(mask, [1, 2])
    nearest_known = ndimage.distance_transform_edt(~known_mask, return_distances=False, return_indices=True)
    filled_mask = mask.copy()
    filled_mask[mask == -1] = mask[tuple(nearest_known[:, mask == -1])]
    
    return filled_mask

def label_3D(a: np.array):
    labs, _ = ndimage.label(a, structure=np.ones((3,3,3)))
    return labs

def label_2D(a: np.array):
    labelled_slices = []
    indices_used = 0
    for i in range(a.shape[2]):
        labs, n = ndimage.label(a[...,i], structure=np.ones((3,3)))
        labs_updated = copy.deepcopy(labs)
        labs_updated += indices_used
        labs_updated[labs==0]=0
        indices_used += n
        labelled_slices.append(labs_updated)
    labs = np.stack(labelled_slices, axis=2)
    return labs


# Add volumes and counts (total, per ROI, per ROI max slice and hemisphere)

path_to_root = Path(__file__).parents[2]
data_dir = path_to_root / "data_new"
labels_dir = data_dir / "labels"
splits_dir = data_dir / "splits"
results_dir = path_to_root / "results_new" / "segmentor" / "dynunet"

def process(df, is_preds):
    vols, counts, BG_vols, BG_counts, CSO_vols, CSO_counts, BG_max_vols, BG_max_counts, CSO_max_vols, CSO_max_counts = [], [], [], [], [], [], [], [], [], []
    for row in tqdm(df.iterrows(), total=len(df)):
        id_ = row[1]["id"]
        if is_preds:
            PVS_path = results_dir / f"{id_}_pred.nii.gz"
        else:
            PVS_path = labels_dir / f"{id_}_PVS.nii.gz"
        if not PVS_path.exists():
            vols.append("N/A")
            counts.append("N/A")
            BG_vols.append("N/A")
            BG_counts.append("N/A")
            CSO_vols.append("N/A")
            CSO_counts.append("N/A")
            BG_max_vols.append("N/A")
            BG_max_counts.append("N/A")
            CSO_max_vols.append("N/A")
            CSO_max_counts.append("N/A")
            continue
        PVS_nii = nib.load(PVS_path)
        PVS = PVS_nii.get_fdata()
        if is_preds:
            PVS = np.where(PVS >= 0.5, 1, 0)
        BG = nib.load(labels_dir / f"{id_}_BG.nii.gz").get_fdata()
        CSO = nib.load(labels_dir / f"{id_}_CSO.nii.gz").get_fdata()
        synthseg = nib.load(labels_dir / f"{id_}_synthseg.nii.gz").get_fdata()
        LR = split_hemispheres(synthseg)
        vox_size = np.prod(PVS_nii.header.get_zooms())
        
        PVS_cc_3D = label_3D(PVS)
        PVS_cc_2D = label_2D(PVS)

        vols.append(np.sum(PVS) * vox_size)
        counts.append(len(np.unique(PVS_cc_3D)[1:]))
        
        BG_vols.append(np.sum(BG * PVS) * vox_size)
        BG_counts.append(len(np.unique(BG * PVS_cc_3D)[1:]))
        
        CSO_vols.append(np.sum(CSO * PVS) * vox_size)
        CSO_counts.append(len(np.unique(CSO * PVS_cc_3D)[1:]))
        
        for roi_mask, roi_vol_list, roi_count_list in [(BG, BG_max_vols, BG_max_counts), (CSO, CSO_max_vols, CSO_max_counts)]:
            m_vol, m_count = 0, 0
            PVS_masked = copy.deepcopy(PVS)
            PVS_cc_2D_masked = copy.deepcopy(PVS_cc_2D)
            PVS_masked[roi_mask!=1]=0
            PVS_cc_2D_masked[roi_mask!=1]=0
            for hem in [1,2]:
                PVS_masked[LR!=hem]=0
                PVS_cc_2D_masked[LR!=hem]=0
                for slc in range(PVS.shape[2]):
                    vol = np.sum(PVS_masked[...,slc]) * vox_size
                    count = len(np.unique(PVS_cc_2D_masked[...,slc])[1:])
                    m_vol = max(m_vol, vol)
                    m_count = max(m_count, count)
            roi_vol_list.append(m_vol)
            roi_count_list.append(m_count)
            
    s = "_gt" if not is_preds else "_pred"
            
    df[f"vols{s}"] = vols
    df[f"counts{s}"] = counts
    df[f"BG_vols{s}"] = BG_vols
    df[f"BG_counts{s}"] = BG_counts
    df[f"CSO_vols{s}"] = CSO_vols
    df[f"CSO_counts{s}"] = CSO_counts
    df[f"BG_max_vols{s}"] = BG_max_vols
    df[f"BG_max_counts{s}"] = BG_max_counts
    df[f"CSO_max_vols{s}"] = CSO_max_vols
    df[f"CSO_max_counts{s}"] = CSO_max_counts
    
    return df


df = pd.read_csv(data_dir / "metadata.csv").fillna("N/A")
# sort by id
df = process(df, is_preds=False)
df = process(df, is_preds=True)

# save as metadata2
df.to_csv(data_dir / "metadata_extra_cols.csv", index=False)

# add the new columns to the subdataframes train, val and test in the ../../data/splits directory. Make the the ids in the 'id' column match

df_train = pd.read_csv(splits_dir / "train.csv").fillna("N/A")
df_val = pd.read_csv(splits_dir / "val.csv").fillna("N/A")
df_test = pd.read_csv(splits_dir / "test.csv").fillna("N/A")

def merge_new_columns(sub_df, full_df):
    new_cols = [col for col in full_df.columns if col not in sub_df.columns]
    return sub_df.merge(full_df[["id"] + new_cols], on="id", how="left")

df_train = merge_new_columns(df_train, df)
df_val = merge_new_columns(df_val, df)
df_test = merge_new_columns(df_test, df)

df_train.to_csv(splits_dir / "train_extra_cols.csv", index=False)
df_val.to_csv(splits_dir / "val_extra_cols.csv", index=False)
df_test.to_csv(splits_dir / "test_extra_cols.csv", index=False)