"""Data loaders for HCP and ABIDE fMRI datasets.

Both datasets live in a single flat directory containing NumPy arrays.

HCP layout  (748 subjects, 379 regions):
    bold.npy         (S, N, T)  — BOLD time series, axes order NT per subject
    connectivity.npy (S, N, N)  — pre-computed Pearson FC matrix
    coords.npy       (N, 3)     — MNI centroid coordinates (shared)
    labels.npy       (S,)       — binary class labels

ABIDE layout (884 subjects, 160 regions):
    bold.npy         (S, T, N)  — BOLD time series, axes order TN per subject
    connectivity.npy (S, N, N)
    coords.npy       (N, 3)
    labels.npy       (S,)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np

# bold_axes tells subject_to_data whether bold is (N,T) or (T,N)
class SubjectRecord(NamedTuple):
    bold: np.ndarray
    fc: np.ndarray
    label: int
    subject_id: str
    bold_axes: str
    site: str | int | None = None


def load_hcp(root: str) -> Tuple[List[SubjectRecord], np.ndarray]:
    """Load HCP dataset from *root*.

    Returns:
        subjects: list of SubjectRecord; bold is (N, T) per subject.
        coords:   (N, 3) MNI coordinates shared across subjects.
    """
    root = Path(root)
    bold_arr = np.load(root / "bold.npy")           # (S, N, T)
    fc_arr   = np.load(root / "connectivity.npy")   # (S, N, N)
    labels   = np.load(root / "labels.npy")         # (S,)
    coords   = np.load(root / "coords.npy")         # (N, 3)

    subjects = [
        SubjectRecord(
            bold=bold_arr[i],
            fc=fc_arr[i],
            label=int(labels[i]),
            subject_id=f"hcp_{i:04d}",
            bold_axes="NT",
        )
        for i in range(len(labels))
    ]
    return subjects, coords


def load_abide(root: str) -> Tuple[List[SubjectRecord], np.ndarray]:
    """Load ABIDE dataset from *root*.

    Returns:
        subjects: list of SubjectRecord; bold is (T, N) per subject
                  (subject_to_data transposes internally via bold_axes="TN").
        coords:   (N, 3) MNI coordinates shared across subjects.
    """
    root = Path(root)
    bold_arr = np.load(root / "bold.npy")           # (S, T, N)
    fc_arr   = np.load(root / "connectivity.npy")   # (S, N, N)
    labels   = np.load(root / "labels.npy")         # (S,)
    coords   = np.load(root / "coords.npy")         # (N, 3)
    sites = _load_optional_abide_sites(root, len(labels))

    subjects = [
        SubjectRecord(
            bold=bold_arr[i],
            fc=fc_arr[i],
            label=int(labels[i]),
            subject_id=f"abide_{i:04d}",
            bold_axes="TN",
            site=None if sites is None else sites[i].item(),
        )
        for i in range(len(labels))
    ]
    return subjects, coords


def _load_optional_abide_sites(root: Path, n_subjects: int) -> np.ndarray | None:
    """Load optional ABIDE acquisition-site labels from common flat files."""
    candidates = (
        "sites.npy",
        "site.npy",
        "site_ids.npy",
        "site_labels.npy",
        "batch.npy",
        "batches.npy",
    )
    for name in candidates:
        path = root / name
        if not path.is_file():
            continue
        sites = np.load(path, allow_pickle=True)
        if sites.shape[0] != n_subjects:
            raise ValueError(
                f"{path} contains {sites.shape[0]} site labels, but ABIDE has "
                f"{n_subjects} subjects"
            )
        return sites
    return None


def load_islem(
    dataset_name: str,
    root: str,
) -> Tuple[List[SubjectRecord], np.ndarray]:
    """Load one of Islem's datasets (ad_lmci or nc_asd).

    File layout (flat directory, shared coords file):
        data_{dataset_name}.npy    (S, N, N, F)  — F connectivity feature maps;
                                                    feature index 0 is used as FC.
        labels_{dataset_name}.npy  (S,)           — binary class labels.
        desikan_coords_left.npy    (N, 3)          — shared MNI centroids.

    Because no BOLD time-series is available, the FC matrix (feature 0, shape
    N×N) is reused as the BOLD signal.  Each region's row in FC becomes its
    T-dimensional time series (T = N), so bold_axes="NT".
    """
    root = Path(root)
    data_arr = np.load(root / f"data_{dataset_name}.npy")   # (S, N, N, F)
    labels   = np.load(root / f"labels_{dataset_name}.npy") # (S,)
    coords   = np.load(root / "desikan_coords_left.npy")    # (N, 3)

    fc_arr   = data_arr[:, :, :, 0]                         # (S, N, N)

    subjects = [
        SubjectRecord(
            bold=fc_arr[i],       # (N, N) — FC used as proxy BOLD
            fc=fc_arr[i],
            label=int(labels[i]),
            subject_id=f"{dataset_name}_{i:04d}",
            bold_axes="NT",       # treat rows as (N_regions, T=N)
        )
        for i in range(len(labels))
    ]
    return subjects, coords


def load_ad_lmci(root: str) -> Tuple[List[SubjectRecord], np.ndarray]:
    return load_islem("ad_lmci", root)


def load_nc_asd(root: str) -> Tuple[List[SubjectRecord], np.ndarray]:
    return load_islem("nc_asd", root)


_LOADERS = {
    "hcp":     load_hcp,
    "abide":   load_abide,
    "ad_lmci": load_ad_lmci,
    "nc_asd":  load_nc_asd,
}


def load_dataset(
    name: str,
    root: str,
) -> Tuple[List[SubjectRecord], np.ndarray]:
    """Dispatch to the appropriate dataset loader.

    Args:
        name: "hcp" or "abide".
        root: path to the flat raw directory.

    Returns:
        (subjects, coords)
    """
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(_LOADERS)}"
        )
    return _LOADERS[name](root)
