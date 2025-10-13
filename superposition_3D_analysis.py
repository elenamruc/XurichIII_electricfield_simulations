# =====================================================
# superposition_plots.py
# Author: Elena Muñoz
#
# This script loads unit field maps (1 V applied to each electrode),
# defines the detector geometry and regions of interest (ROI),
# and provides helper functions to read data and process the
# electric field in cylindrical coordinates (r, z, Er, Ez).
#
# The main goal is to evaluate the field uniformity in the LXe region
# and to determine the optimal electrode voltage configuration
# by combining unit-field bases through superposition
# =====================================================

import numpy as np
import pandas as pd
import re, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Directory containing precomputed 1-V basis field files (HDF5 or CSV format)
BASIS_DIR = "./basis_fields/"
PATH_BASIS     = "/Volumes/NO NAME/superposition_3D_27mm_27mm.csv" 

# Directory for saving output plots
output_path = "./plots/" 
  
# Optional validation case (for comparison with a direct COMSOL field)
PATH_EXAMPLES  = "example_3D_4000.csv"
DO_VALIDATION  = False                      # Set True only to compare with PATH_EXAMPLES (validation mode)

# --- Detector geometry (all distances in meters) ---
D_CG           = 30.8e-3                   # cathode-gate distance (m) for analytical seed
d_cathode      = -0.0289/2 - 0.002         # cathode z-position (m)
d_gate         = +0.0289/2                 # gate z-position (m)
d_anode        = +0.0289/2 + 0.0041        # anode z-position (m)

# --- ROI definitions (meters) ---
R_DET_M        = (31.0/2.0) * 1e-3         # Radial detector extent
#ROI_R          = (0, (31.0/2.0) * 1e-3)   # Radial range of interest for analysis
ROI_R          = (0, 10.0 * 1e-3)
MARGIN_Z       = 2e-3                      # Safety margin to avoid boundary artifacts near electrodes

# LXe region: between cathode and gate
ROI_LXe        = (ROI_R[0], ROI_R[1], d_cathode + MARGIN_Z, d_gate - MARGIN_Z)

# GXe region: between gate and anode (used for extraction-field study)
ROI_GXe        = (ROI_R[0], ROI_R[1], d_gate + MARGIN_Z, d_anode - 5e-4)

# --- ROIs used for penalty terms in optimization ---
# Below-cathode band (captures field reversal and PMT hotspot region)
ROI_BELOW_CATH = (ROI_R[0], ROI_R[1], d_cathode - 6e-3, d_cathode - 1e-3)

# Near-gate band (captures field leakage from extraction region)
ROI_TOP_LXE    = (ROI_R[0], ROI_R[1], d_gate - 1.0e-3,  d_gate - 3.0e-4)


# --- Fixed hardware voltages ---
EXTRACTION = 4000.0                       # Gate–Anode potential difference (V)
PMT_TOP_V  = -1000.0
PMT_BOT_V  = -1000.0

# --- HV limits (for optimization constraints) ---
ANODE_MAX = +5000.0                       # maximum anode voltage (V)
GATE_MAX_BY_ANODE = ANODE_MAX - EXTRACTION

# Voltage bounds for each controllable electrode (used in optimization)
BOUNDS = dict(
    gate=(-GATE_MAX_BY_ANODE, GATE_MAX_BY_ANODE),
    cath=(-5000.0, 2000.0),
    pmt_top=(-1100.0, -900.0),
    pmt_bot=(-1100.0, -900.0)
)

def basis_support_mask(Bstack, thresh=1e-9):
    """
    Returns a boolean mask identifying the spatial region
    where the total field magnitude from all basis components
    is significant (> thresh).
    """
    Er = Bstack['Er']; Ez = Bstack['Ez']

    # Compute the total norm for all basis fields with 1 V applied
    S = np.sum(np.hypot(Er, Ez), axis=0)   # (Nz, Nr)
    return S > float(thresh)


# Mapping from basis index (as stored in COMSOL output) to electrode name
IDX2NAME = {
    1: 'cath',
    2: 'gate',
    3: 'anode',
    4: 'pmt_top',
    5: 'pmt_bot',
    6: 'ring_1',
    7: 'ring_2',
    8: 'ring_3',
    9: 'ring_4',
    10:'ring_5',
    11:'ring_6',
    12:'ring_7',
}

# Z-positions of key electrodes (for plotting / visualization)
Z_CATH = -15.95e-3
Z_GATE = +14.05e-3

# Z positions of field-shaping rings (from bottom to top)
RING_Z = {
    'ring_7': -12.45e-3,
    'ring_6': -8.45e-3,
    'ring_5': -4.45e-3,
    'ring_4': -0.45e-3,
    'ring_3':  +3.55e-3,
    'ring_2':  +7.55e-3,
    'ring_1': +11.55e-3,   # near gate
}

# Step between rings (in mm) — used for auto-generation if needed
_step = 24.0/7.0  # mm

# Manual list of ring positions considering another RING: 
_ring_mm = [
    -12.45,                               # ring_8 (closest to cathode)
    -12.45 + _step,                       # ring_7
    -12.45 + 2*_step,                     # ring_6
    -12.45 + 3*_step,                     # ring_5
    -12.45 + 4*_step,                     # ring_4
    -12.45 + 5*_step,                     # ring_3
    -12.45 + 6*_step,                     # ring_2
    +11.55                                # ring_1 (near gate)
]

#RING_Z = {
#    'ring_8': _ring_mm[0]*1e-3,
#    'ring_7': _ring_mm[1]*1e-3,
#    'ring_6': _ring_mm[2]*1e-3,
#    'ring_5': _ring_mm[3]*1e-3,
#    'ring_4': _ring_mm[4]*1e-3,
#    'ring_3': _ring_mm[5]*1e-3,
#    'ring_2': _ring_mm[6]*1e-3,
#    'ring_1': _ring_mm[7]*1e-3,
#}



# Radial binning settings
NR_BINS = 100                 # Number of radial bins (used for rebinning E-field)
RMAX_FOR_BIN = ROI_R[1]       # Maximum radius for analysis

def _auto_cache_name(path_csv, Nr, rmax):
    """
    Helper to generate a cache filename automatically based on
    the original basis file, number of radial bins, and rmax (in mm).
    """
    base = os.path.splitext(os.path.basename(path_csv))[0]
    rmm = int(round(rmax * 1e3))
    return f"basis_cache_{base}_Nr{Nr}_R{rmm}mm.npz"


# Cached version of preprocessed data
CACHE_NPZ = _auto_cache_name(PATH_BASIS, NR_BINS, RMAX_FOR_BIN)

# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def read_with_commented_header(path):
    """
    Reads a CSV file ignoring lines starting with '%'.
    If a commented header containing field names is found,
    it will be used as column names.
    Accepts any prefix before 'Ex/Ey/Ez' (e.g., 'es2.Ex (V/m)').
    """
    names = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s.startswith('%') and ',' in s and ('Ex (V/m)' in s or 'Ey (V/m)' in s or 'Ez (V/m)' in s):
                # Extract column names from the commented header
                names = [t.strip().lstrip('%').strip() for t in s.split(',')]
                break
    if names:
        return pd.read_csv(path, comment='%', header=None, names=names, engine='python')
    else:
        return pd.read_csv(path, comment='%', engine='python')

def _pick(df, *cands):
    """
    Returns the first column name found in DataFrame `df`
    that matches any of the candidate names in `cands`.
    """
    for c in cands:
        if c in df.columns:
            return c
    return None

def _pick_field_vcomp(df, which='Ex'):
    """
    Finds a column name containing a specific E-field component (Ex, Ey, Ez),
    even if the name includes a COMSOL prefix, e.g., 'es2.Ex (V/m)'.

    Returns the column name or None if not found.
    """
    tail = f"{which} (V/m)"
    for col in df.columns:
        if str(col).strip().endswith(tail):
            return col
    # Regex match (for non-standard names)
    pat = re.compile(rf"(^|[.\s_]){which}($|[\s(])", re.IGNORECASE)
    for col in df.columns:
        if pat.search(str(col)):
            return col
    return None

def _to_m_if_mm(arr_like):
    """
    Converts an array from mm to m if typical values are larger than 0.5,
    assuming they are in mm or cm. Otherwise, leaves values unchanged.
    """
    a = np.asarray(arr_like, dtype=float)
    if np.nanpercentile(np.abs(a), 90) > 0.5:
        return a / 1000.0
    return a

def _unit_phi(x, y):
    """
    Given Cartesian coordinates (x, y), returns
    (cos(phi), sin(phi), phi) where phi is the azimuthal angle.
    """
    phi = np.arctan2(y, x)
    return np.cos(phi), np.sin(phi), phi

def _edges_uniform_r2(rmax, Nr):
    """
    Returns bin edges for a uniform binning in r² from 0 to rmax².
    """
    return np.sqrt(np.linspace(0.0, rmax**2, Nr+1))

def _bin_to_r_grid(df_slice, Nr, rmax):
    """
    Performs radial binning at a fixed z position.
    Uses uniform binning in r² to avoid undersampling near the center.
    """
    edges = _edges_uniform_r2(rmax, Nr)
    idx = np.clip(np.digitize(df_slice['r'].values, edges) - 1, 0, Nr-1)
    Er_b = np.zeros(Nr); Ez_b = np.zeros(Nr); cnt = np.zeros(Nr, dtype=int)
    np.add.at(Er_b, idx, df_slice['Er'].values)
    np.add.at(Ez_b, idx, df_slice['Ez'].values)
    np.add.at(cnt,   idx, 1)
    cnt = np.where(cnt==0, 1, cnt)
    Er_b /= cnt
    Ez_b /= cnt
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return r_centers, Er_b, Ez_b

def median_signed_Ez(Ez, mask):
    """
    Computes the median value of Ez inside a given mask,
    excluding NaNs and non-finite entries.
    """
    vals = Ez[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else 0.0

# ============================================================
# 3. BASIS AND DIRECT FIELD LOADERS
# ============================================================

def load_basis_cartesian(path, Nr=NR_BINS, rmax=RMAX_FOR_BIN):
    """
    Loads a COMSOL 'unit-field' CSV (1 V applied to each electrode),
    converts it to cylindrical coordinates (r, Er, Ez),
    and rebins data into a uniform radial grid (uniform in r²).

    The output is organized by electrode name and can be used
    to reconstruct the total electric field by linear superposition.
    """
    df = read_with_commented_header(path).rename(columns=lambda s: str(s).strip())

    # geometry columns
    xcol = _pick(df, 'x (mm)', 'x'); ycol = _pick(df, 'y (mm)', 'y'); zcol = _pick(df, 'z (mm)', 'z')

    # field columns (accepts prefixes like 'es2.Ex (V/m)')
    Ex = _pick_field_vcomp(df, 'Ex'); Ey = _pick_field_vcomp(df, 'Ey'); Ez = _pick_field_vcomp(df, 'Ez')

    # index column (which 1-V electrode)
    idxcol = _pick(df, 'idx', 'Parameter value')

    assert all([xcol, ycol, zcol, Ex, Ey, Ez, idxcol]), "Missing columns in basis CSV."

    # to float
    df['x']  = df[xcol].astype(float)
    df['y']  = df[ycol].astype(float)
    df['z']  = df[zcol].astype(float)
    df['Ex'] = df[Ex].astype(float)
    df['Ey'] = df[Ey].astype(float)
    df['Ez'] = df[Ez].astype(float)
    df = df.dropna(subset=['Ex','Ey','Ez'])

    # mm → m if needed
    df['x'] = _to_m_if_mm(df['x'].values)
    df['y'] = _to_m_if_mm(df['y'].values)
    df['z'] = _to_m_if_mm(df['z'].values)

    # map 'idx' to electrode names (robust to float idx)
    unknown = set(map(int, pd.unique(df[idxcol]))) - set(IDX2NAME.keys())
    if unknown:
        raise ValueError(f"IDX without mapping in IDX2NAME: {sorted(unknown)}")
    df['electrode'] = df[idxcol].astype(int).map(IDX2NAME)

    # cylindrical coordinates and radial field
    c, s, _ = _unit_phi(df['x'].values, df['y'].values)
    df['r']  = np.sqrt(df['x'].values**2 + df['y'].values**2)
    df['Er'] = df['Ex'].values*c + df['Ey'].values*s

    # common grids
    z_levels = np.sort(df['z'].unique())
    # r-grid directly from edges (uniform in r^2), independent of data density
    r_edges = _edges_uniform_r2(rmax, Nr)
    r_grid  = 0.5 * (r_edges[:-1] + r_edges[1:])

    out = {}
    # ensure deterministic electrode order
    for name in sorted(df['electrode'].unique(), key=lambda x: (x is None, str(x))):
        sub_all = df[df['electrode'] == name]
        Er_mat = np.zeros((len(z_levels), Nr), dtype=np.float32)
        Ez_mat = np.zeros_like(Er_mat)

        for i, z0 in enumerate(z_levels):
            sli = sub_all[sub_all['z'] == z0][['r', 'Er', 'Ez']]
            if sli.empty:
                # fill zeros (field from this electrode at this z is unavailable)
                Er_mat[i, :] = 0.0
                Ez_mat[i, :] = 0.0
                continue
            # keep only r<=rmax
            sli = sli[sli['r'] <= rmax]
            if sli.empty:
                Er_mat[i, :] = 0.0
                Ez_mat[i, :] = 0.0
                continue
            # bin to common r-grid (uniform in r^2)
            _, Er_b, Ez_b = _bin_to_r_grid(sli, Nr, rmax)
            Er_mat[i, :] = Er_b.astype(np.float32)
            Ez_mat[i, :] = Ez_b.astype(np.float32)

        out[name] = dict(r=r_grid.astype(np.float32),
                         z=z_levels.astype(np.float32),
                         Er=Er_mat, Ez=Ez_mat)

    return out

# ============================================================
# 4. DIRECT FIELD LOADERS (from full-bias COMSOL simulations)
# ============================================================

from scipy.interpolate import griddata

def load_direct_on_basis_grid_interp(path, Bstack, rmax=RMAX_FOR_BIN):
    """
    Loads a direct COMSOL CSV (all electrodes biased simultaneously),
    converts to cylindrical coordinates, and interpolates (Er, Ez)
    onto the r-z grid defined by the basis stack (Bstack).
    """
    df = read_with_commented_header(path).rename(columns=lambda s: str(s).strip())

    # Identify required columns
    xcol = _pick(df, 'x (mm)', 'x')
    ycol = _pick(df, 'y (mm)', 'y')
    zcol = _pick(df, 'z (mm)', 'z')
    Ex   = _pick_field_vcomp(df, 'Ex')
    Ey   = _pick_field_vcomp(df, 'Ey')
    Ez   = _pick_field_vcomp(df, 'Ez')
    assert all([xcol, ycol, zcol, Ex, Ey, Ez]), "Missing columns in direct CSV."

    # Convert and sanitize numeric data
    x = _to_m_if_mm(df[xcol].astype(float).values)
    y = _to_m_if_mm(df[ycol].astype(float).values)
    z = _to_m_if_mm(df[zcol].astype(float).values)
    ex = df[Ex].astype(float).values
    ey = df[Ey].astype(float).values
    ez = df[Ez].astype(float).values

    # Cylindrical transformation
    phi = np.arctan2(y, x); c = np.cos(phi); s = np.sin(phi)
    r  = np.sqrt(x*x + y*y)
    er = ex*c + ey*s

    # Keep only finite and valid domain points
    m = np.isfinite(r) & np.isfinite(z) & np.isfinite(er) & np.isfinite(ez) & (r <= rmax)
    r, z, er, ez = r[m], z[m], er[m], ez[m]

    # Target (r,z) grid from basis
    r_grid = np.asarray(Bstack['r'], dtype=float)
    z_grid = np.asarray(Bstack['z'], dtype=float)
    Rg, Zg = np.meshgrid(r_grid, z_grid)  # (Nz, Nr)

    pts = np.column_stack([r, z])

    # Interpolate Er, Ez using linear and nearest methods
    Er_lin = griddata(pts, er, (Rg, Zg), method='linear')
    Ez_lin = griddata(pts, ez, (Rg, Zg), method='linear')
    Er_near = griddata(pts, er, (Rg, Zg), method='nearest')
    Ez_near = griddata(pts, ez, (Rg, Zg), method='nearest')

    # Fill NaNs from linear interpolation with nearest-neighbor values
    Er = np.where(np.isfinite(Er_lin), Er_lin, Er_near).astype(np.float32)
    Ez = np.where(np.isfinite(Ez_lin), Ez_lin, Ez_near).astype(np.float32)

    # Diagnostic printout
    lin_cov = np.isfinite(Er_lin).mean()
    print(f"[INFO] Direct-basis interpolation: linear coverage {100*lin_cov:.1f}% (rest filled with nearest).")

    return dict(r=r_grid.astype(np.float32), z=z_grid.astype(np.float32), Er=Er, Ez=Ez)


# ============================================================
# 5. CACHE AND FAST FIELD SUPERPOSITION
# ============================================================

CACHE_VERSION = 2

def build_basis_cache(path_csv, Nr=NR_BINS, rmax=RMAX_FOR_BIN, cache=CACHE_NPZ):
    """
    Builds and saves a compressed cache (.npz) of all basis field maps.
    The cache speeds up future reloads by storing the r,z grids and
    all (Er, Ez) matrices in binary format.
    """
    basis = load_basis_cartesian(path_csv, Nr=Nr, rmax=rmax)
    names = sorted(basis.keys())
    r = basis[names[0]]['r'].astype(np.float32)
    z = basis[names[0]]['z'].astype(np.float32)
    Er = np.stack([basis[n]['Er'] for n in names]).astype(np.float32)  # (Ne, Nz, Nr)
    Ez = np.stack([basis[n]['Ez'] for n in names]).astype(np.float32)

    meta = dict(
        version=CACHE_VERSION,
        path=os.path.abspath(path_csv),
        mtime=os.path.getmtime(path_csv) if os.path.exists(path_csv) else None,
        size=os.path.getsize(path_csv) if os.path.exists(path_csv) else None,
        Nr=int(Nr),
        rmax=float(rmax),
        names=list(names),
    )
    # np.savez_compressed cannot store dicts directly, so meta is stored as an object
    np.savez_compressed(cache,
                        names=np.array(names),
                        r=r, z=z, Er=Er, Ez=Ez,
                        meta=np.array([meta], dtype=object))
    return dict(names=names, r=r, z=z, Er=Er, Ez=Ez)

def load_basis_cache(path_csv, Nr=NR_BINS, rmax=RMAX_FOR_BIN, cache=CACHE_NPZ, force_rebuild=False, trust_cache=True):
    """
    Loads the cached basis file if valid; otherwise rebuilds it.
    Metadata are checked (version, Nr, rmax, file size, modification time)
    to ensure consistency with the original CSV.
    """
    def _ok_meta(npz):
        if 'meta' not in npz.files:
            return False
        meta = npz['meta'].item()
        if meta.get('version') != CACHE_VERSION:
            return False
        # Consistency checks
        if int(meta.get('Nr', -1)) != int(Nr):
            return False
        if abs(float(meta.get('rmax', -1.0)) - float(rmax)) > 1e-9:
            return False
        # Compare with CSV source file
        if os.path.exists(path_csv):
            try:
                if abs(meta.get('mtime', -1.0) - os.path.getmtime(path_csv)) > 1e-6:
                    return False
                if int(meta.get('size', -1)) != os.path.getsize(path_csv):
                    return False
            except Exception:
                return False
        return True

    if (not force_rebuild) and os.path.exists(cache):
        npz = np.load(cache, allow_pickle=True, mmap_mode='r')
        if trust_cache or _ok_meta(npz):   
            # Return cached data directly
            return dict(names=list(npz['names']), r=npz['r'], z=npz['z'],
                        Er=npz['Er'], Ez=npz['Ez'])
        else:
            # Rebuild if cache is outdated
            try: 
                os.remove(cache)
            except Exception: 
                pass

    return build_basis_cache(path_csv, Nr=Nr, rmax=rmax, cache=cache)

def superpose_fast(Bstack, Vdict):
    """
    Compute the total electric field by superposing all unit-field maps
    according to the voltage configuration provided in Vdict.
    """
    names = Bstack['names']
    weights = np.array([Vdict.get(n, 0.0) for n in names], dtype=np.float32)  # (Ne,)
    Er_total = np.tensordot(weights, Bstack['Er'], axes=1)  # (Nz, Nr)
    Ez_total = np.tensordot(weights, Bstack['Ez'], axes=1)  # (Nz, Nr)
    return Bstack['r'], Bstack['z'], Er_total, Ez_total


# ============================================================
# 6. ROI, FIELD METRICS AND VOLTAGE SETUP
# ============================================================

def roi_mask(r, z, rmin, rmax, zmin, zmax):
    """
    Generate a boolean mask for points within a cylindrical ROI.
    """
    R, Z = np.meshgrid(r, z)
    return (R >= rmin) & (R <= rmax) & (Z >= zmin) & (Z <= zmax)


def list_rings_from_basis(basis_like):
    """
    Returns the list of ring electrode names (ring_1, ring_2, ...)
    sorted in ascending numerical order.
    """
    if isinstance(basis_like, dict) and 'names' in basis_like:
        names = basis_like['names']
    else:
        names = list(basis_like.keys())
    rings = sorted([k for k in names if str(k).startswith('ring_')], key=lambda s: int(str(s).split('_')[1]))
    return rings

def Vdict_for_E(gate_voltage=0.0, pmt_top=None, pmt_bottom=None,
                include_rings=True, basis_like=None, ring_positions=None):
    """
    Build a complete voltage configuration dictionary for all electrodes.

    The resulting dictionary can be directly passed to `superpose_fast`
    to compute the total electric field for a given configuration.
    """
    cathode_voltage = -4000.0
    anode_voltage   = gate_voltage + EXTRACTION
    V = dict(gate=gate_voltage, cath=cathode_voltage, anode=anode_voltage)

    # Optional PMT voltages
    if pmt_top is not None:
        V['pmt_top'] = pmt_top
    if pmt_bottom is not None:
        V['pmt_bot'] = pmt_bottom

    # Add voltages for field-shaping rings
    if include_rings and basis_like is not None:
        ring_names = list_rings_from_basis(basis_like)
        if ring_positions is not None:
            # Use actual z positions to interpolate between cathode and gate
            V.update(rings_from_positions(cathode_voltage, gate_voltage,
                                          ring_names, ring_positions,
                                          z_cath=Z_CATH, z_gate=Z_GATE))
        else:
            # Fallback: use a simple uniform ladder (only valid for ideal ring spacing)
            n = len(ring_names)
            step = (gate_voltage - cathode_voltage) / (n + 1) if n > 0 else 0.0
            for i, name in enumerate(ring_names, start=1):
                # ring_1 is closest to the gate
                i_rev = (n + 1) - i
                V[name] = cathode_voltage + i_rev * step
    return V

def rings_from_positions(Vcath, Vgate, ring_names, ring_z, z_cath=Z_CATH, z_gate=Z_GATE):
    """
    Interpolate ring voltages between cathode and gate
    based on their physical z positions.
    """
    L = float(z_gate - z_cath)
    out = {}

    # Sort by position (bottom -> top)
    names_sorted = sorted(ring_names, key=lambda n: ring_z[n])
    for name in names_sorted:
        zi = float(ring_z[name])
        t = (zi - z_cath) / L        # Normalized 0..1 coordinate between cathode and gate
        out[name] = Vcath + t * (Vgate - Vcath)
    return out


def segment_weights_from_positions(ring_names, ring_z, z_cath=Z_CATH, z_gate=Z_GATE):
    """
    Compute relative segment lengths (weights) between cathode, rings, and gate.
    """
    names_bottom_up = sorted(ring_names, key=lambda n: ring_z[n])  # increasing z
    zs = [z_cath] + [ring_z[n] for n in names_bottom_up] + [z_gate]
    w = np.diff(zs)  # (n+1,)
    return names_bottom_up, np.asarray(w, float)

def rings_from_segment_weights(Vc, Vg, names_bottom_up, w):
    """
    Compute ring voltages from relative segment weights.
    """
    w = np.clip(np.asarray(w, float), 1e-12, None)
    t_nodes = np.cumsum(w) / np.sum(w)        # normalized fraction 0..1 per node          
    t_rings = t_nodes[:-1]                    # exclude final 'gate' node              
    out = {}
    for name, t in zip(names_bottom_up, t_rings):
        out[name] = Vc + t * (Vg - Vc)
    return out


def ring_weights_uniform_plus_top(ring_names, top_factor=1.0):
    """
    Create a set of segment weights for `rings_from_positions`,
    where all segments are uniform (=1) except the one just below the gate
    (ring_2 → ring_1), which is scaled by `top_factor`.
    """
    n = len(ring_names)
    if n == 0:
        return np.array([1.0], float)

    w = np.ones(n + 1, dtype=float)   # n segments + 1 up to gate
    w[n - 1] = max(1e-4, float(top_factor))  # adjust second-to-last segment
    return w


# ============================================================
# 7. FIELD METRICS AND QUALITY INDICATORS
# ============================================================

def median_gas_field_from_V(Bstack, V, roi_gas):
    """
    Compute the median |Ez| field in the gas (GXe) region for a given voltage setup.
    The result is returned in kV/cm.
    """
    r, z, Er, Ez = superpose_fast(Bstack, V)
    Mgas = roi_mask(r, z, *roi_gas)
    Ez_abs = np.abs(Ez[Mgas])
    return (np.nanmedian(Ez_abs) / 1e5) if Ez_abs.size else np.nan


def frac_reversed(Ez, mask, expected_sign=-1):
    """
    Fraction of points within the ROI where Ez has the opposite sign
    to the expected drift direction (indicates field reversal).
    """
    idx = np.where(mask)
    if idx[0].size == 0: return 0.0
    vals = Ez[idx]; vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0
    return float(np.mean(np.sign(vals) != expected_sign))


# ============================================================
# 8. PLOTTING FUNCTIONS
# ============================================================

def plot_diff_map(Bstack, Vdict, direct, roi_lxe=None, full_extent=False,
                  r2_max_cm2=None, out='diff_map.png'):
    """
    Plot the relative deviation map between superposed and direct COMSOL fields:
    |E_superposed - E_direct| / |E_direct| on the (r², z) plane.
    """
    r, z, Er, Ez = superpose_fast(Bstack, Vdict)
    Er_d, Ez_d   = direct['Er'], direct['Ez']

    # Compute relative error
    num = np.sqrt((Er - Er_d)**2 + (Ez - Ez_d)**2)
    den = np.sqrt(Er_d**2 + Ez_d**2)

    valid = np.isfinite(num) & np.isfinite(den)
    den_med = np.nanmedian(den[valid & (den > 0)])
    eps = max(1e-12, 1e-3*(den_med if np.isfinite(den_med) else 1.0))

    Q = np.full_like(den, np.nan, dtype=float)
    keep = valid & (den > eps)
    Q[keep] = num[keep] / den[keep]

    # Optionally crop to ROI for display
    r_show, z_show, Q_show = r, z, Q
    if (roi_lxe is not None) and (not full_extent):
        rmin, rmax, zmin, zmax = roi_lxe
        R, Z = np.meshgrid(r, z)
        M = (R>=rmin)&(R<=rmax)&(Z>=zmin)&(Z<=zmax)
        rows = np.where(np.any(M, axis=1))[0]
        cols = np.where(np.any(M, axis=0))[0]
        z_show = z[rows]; r_show = r[cols]; Q_show = Q[np.ix_(rows, cols)]

    # Optional crop by r² max (in cm²)
    if r2_max_cm2 is not None:
        r_cut = (r2_max_cm2**0.5)/100.0
        keep_r = r_show <= r_cut
        r_show = r_show[keep_r]
        Q_show = Q_show[:, keep_r]

    # Build plot extent in (r² [cm²], z [cm])
    R2 = (np.meshgrid(r_show, z_show)[0]**2) * 1e4
    extent = [np.nanmin(R2), np.nanmax(R2), np.nanmin(z_show)*100, np.nanmax(z_show)*100]

    print("rel.err stats: min=", np.nanmin(Q_show), "med=", np.nanmedian(Q_show), "max=", np.nanmax(Q_show))

    plt.figure(figsize=(6, 3.8))
    im = plt.imshow(Q_show, origin='lower', extent=extent, aspect='auto',
                    norm=LogNorm(vmin=1e-4, vmax=2e-1))
    im.cmap.set_bad(im.cmap(0.0), 1.0)

     # Draw dashed lines for the LXe ROI
    if roi_lxe is not None:
        _, _, zmin, zmax = roi_lxe
        plt.hlines([zmin*100, zmax*100], extent[0], extent[1],
                   colors='w', linestyles='--', linewidth=1.2)

    cbar = plt.colorbar(im)
    cbar.set_label(r'$|\Delta E|/|E_{\rm direct}|$')
    plt.xlabel(r'$r^2$ (cm$^2$)')
    plt.ylabel('z (cm)')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_optimal(df, out='optimal_voltages_vs_E_3D_2.png'):
    """
    Plot optimal (Vcath, Vgate, Vanode) vs. target median drift field,
    including optional median extraction field in GXe (Egas_kVcm).
    """
    fig, ax = plt.subplots(figsize=(5.9, 3.9))
    ax.plot(df['Ecm'], df['V_cath']/1e3, label=r'$U_{\mathrm{cathode}}$')
    ax.plot(df['Ecm'], df['V_gate']/1e3,  label=r'$U_{\mathrm{gate}}$')
    ax.plot(df['Ecm'], df['V_anode']/1e3, label=r'$U_{\mathrm{anode}}$')
    ax.set_xlabel('Median drift field [V/cm]')
    ax.set_ylabel('Electrode voltage [kV]')

    if 'Egas_kVcm' in df.columns and np.any(np.isfinite(df['Egas_kVcm'])):
        ax2 = ax.twinx()
        ax2.plot(df['Ecm'], df['Egas_kVcm'], ls='--', color='red')
        ax2.set_ylabel('Extraction field [kV/cm]', color='red')
        ax2.tick_params(axis='y', labelcolor='red')


    fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.98))
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_deviation(df, out='deviation_vs_E_3D_2.png'):
    """
    Plot deviation metrics (mean of |Er/Ez|) vs. median drift field.
    Useful to visualize field uniformity across drift region.
    """
    ycol = 'mean' if 'mean' in df.columns else None
    if ycol is None:
        print("[WARN] plot_deviation: no 'mean' or 'mean' found in DataFrame; skipping plot.")
        return
    plt.figure(figsize=(5.8, 3.6))
    plt.semilogy(df['Ecm'], 100*df[ycol], label='mean: $|E_r|/|E_z|$ (LXe)')
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Deviation [%]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_deflection(df, out='deflection_vs_E.png'):
    """
    Plot drift deflection [mm] vs. median drift field.
    """
    if 'deflect_mean_mm' not in df.columns: return
    plt.figure(figsize=(5.8, 3.6))
    plt.plot(df['Ecm'], df['deflect_mean_mm'])
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Drift deflection mean [mm]')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


# ============================================================
# 9. OPTIMIZATION (Single Source of Truth)
# ============================================================

import numpy as np
from scipy.optimize import minimize

def drift_metrics_rel(Er, Ez, mask):
    """
    Compute integral field metrics within a ROI mask.
    """
    idx = np.where(mask)
    if idx[0].size == 0:
        return 0.0, 1e6, 1e6

    Ez_sel = Ez[idx]; Er_sel = Er[idx]
    finite = np.isfinite(Ez_sel) & np.isfinite(Er_sel)
    Ez_sel = Ez_sel[finite]; Er_sel = Er_sel[finite]
    if Ez_sel.size == 0:
        return 0.0, 1e6, 1e6

    Ez_abs = np.abs(Ez_sel); Er_abs = np.abs(Er_sel)
    Ed_med = float(np.median(Ez_abs))

    # Numerical floor to prevent division by tiny Ez
    floor  = max(0.05*Ed_med, float(np.percentile(Ez_abs,10)), 1.0)
    ratio  = Er_abs / np.clip(Ez_abs, floor, None)
    ratio_mean = float(np.nanmean(ratio))

    # RMS of Ez relative to median (field homogeneity indicator)
    rel = (Ez_abs - Ed_med) / max(Ed_med, 1.0)
    inh_rms = float(np.sqrt(np.nanmean(rel**2)))

    return Ed_med, ratio_mean, inh_rms


def solve_cathode_for_E0(
    Bstack, Vg, extraction_V, mask_lxe, E0_Vm,
    base_V, ring_names, vmin, vmax,
    maxit=30, tol=50.0, ring_deltas=None,
    drift_sign=-1,          # -1 → Ez < 0 (downward drift in LXe)
    enforce_order=True,     # ensure Vc < Vg (physical ordering)
    eps_order=10.0,         # minimal voltage margin [V]
    enforce_sign=True,      # avoid Ez sign inversion in ROI
    eps_sign=25.0           # min |Ez| [V/m] to consider valid sign
):
    """
    Solve for cathode voltage (Vc) that achieves a target median field (E0_Vm)
    in the liquid xenon region (LXe).
    """
    target = drift_sign * E0_Vm

    def Ed_of(Vc):
        """Helper: compute median signed Ez for given Vc."""
        V = dict(base_V)
        V['gate']  = Vg
        V['cath']  = Vc
        V['anode'] = Vg + extraction_V

        # Ring ladder configuration (segment-weighted or positional)
        if (ring_deltas is not None) and ring_names:
            names_bu, w_pos = segment_weights_from_positions(ring_names, RING_Z, z_cath=Z_CATH, z_gate=Z_GATE)
         
            w_use = np.asarray(ring_deltas, float)
            if w_use.shape[0] != w_pos.shape[0]:
                # If mismatch → fallback to geometric spacing
                w_use = w_pos
            V.update(rings_from_segment_weights(Vc, Vg, names_bu, w_use))
        else:
           
            V.update(rings_from_positions(Vc, Vg, ring_names, RING_Z, z_cath=Z_CATH, z_gate=Z_GATE))

        _, _, Er, Ez = superpose_fast(Bstack, V)
        return median_signed_Ez(Ez, mask_lxe)


    # Restrict upper bound if enforcing physical order
    if enforce_order:
        vmax = min(vmax, Vg - eps_order)

    # Evaluate field at search bounds
    Ea, Eb = Ed_of(vmin), Ed_of(vmax)

    # If target lies outside range, pick the nearest boundary
    if not (min(Ea, Eb) <= target <= max(Ea, Eb)):
        Vc_best = vmin if abs(Ea - target) < abs(Eb - target) else vmax
        Ed_best = Ed_of(Vc_best)
        hit = True
        # If sign mismatch, adjust toward physically valid side
        if enforce_sign and np.sign(Ed_best) != np.sign(target):
            Vc_best = vmax  # con orden físico ya es el lado “menos campo”
            Ed_best = Ed_of(Vc_best)
        return Vc_best, hit, Ed_best

    # --- Bisection solver ---
    a, b, Ea, Eb = vmin, vmax, Ea, Eb
    for _ in range(maxit):
        m = 0.5*(a+b)
        Em = Ed_of(m)
        if abs(Em - target) < tol:
            hit = False
            if enforce_sign and (np.sign(Em) != np.sign(target) or abs(Em) < eps_sign):
                hit = True
            return m, hit, Em
        # Update search interval
        if (Ea <= target <= Em) or (Em <= target <= Ea):
            b, Eb = m, Em
        else:
            a, Ea = m, Em

    # Return final midpoint after max iterations
    m = 0.5*(a+b); Em = Ed_of(m)
    hit = abs(Em - target) >= tol
    if enforce_sign and (np.sign(Em) != np.sign(target) or abs(Em) < eps_sign):
        hit = True
    return m, hit, Em

def ring_weights_from_power_alpha(ring_names, alpha):
    """
    Generate segment weights (n_rings+1) following a power-law spacing
    controlled by exponent `alpha`. For alpha>1, rings cluster near the gate.
    """
    n = len(ring_names)
    if n == 0:
        return np.array([1.0], float)
    t = np.array([(n + 1 - i) / (n + 1) for i in range(1, n + 1)], float)
    t = t ** float(alpha)
    # pesos = diferencias de los t ordenados hacia el gate, más el tramo final hasta gate
    w = np.diff(np.r_[0.0, t[::-1], 1.0])
    return np.clip(w, 1e-6, None)

def optimize_voltages_constrained(
    Bstack, E_targets_Vcm, roi_lxe,
    opt_pmts=False, opt_rings=False,
    lambda_inh=0.6, lambda_smooth=0.02,
    opt_offset=False, delta_bounds=(-1000.0, 1000.0),
    # --- novedades útiles ---
    vg_seeds=(-2000.0, -1000.0, -300.0, -150.0, 0.0, +150.0, +300.0, +1000.0, 2000.0),
    leak_band_mm=(2.0, 0.2),      
    w_leak=0.05, w_hot=0.01, w_rev=0.01,  
    order_margin_V=0.0, metric='angle_bal', rings_top_factor= None, rings_power_alpha=None         
):
    """
    Multi-start constrained voltage optimization using L-BFGS-B.

    Strategy
    --------
    - For each target drift field E0, V_cath is solved analytically
      (via bisection) to satisfy median(Ez) ≈ target.
    - Gate voltage (and optionally PMTs/rings) are optimized numerically.
    - Multiple gate voltage seeds are tested to avoid local minima.

    Objective
    ---------
    Minimize:
        mean(|Er|/|Ez|) + λ_inh * RMS(Ez/median(Ez))
    plus additional penalties for:
        - field reversals (wrong Ez sign)
        - high-voltage constraint violations
        - deviation from target E0
    Returns
    -------
    DataFrame with optimal voltages and metrics for each E_target.

    Notes
    -----
    - Physically enforces Vc < Vg < Vanode.
    - All voltages are in [V]; fields in [V/cm].
    """

    r, z = Bstack['r'], Bstack['z']
    M_LXE = roi_mask(r, z, *roi_lxe)

    # Define auxiliary ROIs for leakage and hotspot evaluation
    dz_hi, dz_lo = leak_band_mm[0]*1e-3, leak_band_mm[1]*1e-3
    M_LXE  = roi_mask(Bstack['r'], Bstack['z'], *ROI_LXe)  & SUPPORT
    ROI_TOP_WIDE = (ROI_R[0], ROI_R[1], d_gate - dz_hi, d_gate - dz_lo)
    M_TOP = roi_mask(r, z, *ROI_TOP_WIDE) & SUPPORT
    M_BELOW = roi_mask(r, z, *ROI_BELOW_CATH) & SUPPORT

    if not np.any(M_TOP):
        print("[WARN] Leak band does not intersect mesh; increase 'leak_band_mm'.")
    if not np.any(M_BELOW):
        print("[WARN] Below-cathode band missing; check ROI_BELOW_CATH configuration.")


    ring_names = list_rings_from_basis(Bstack); nR = len(ring_names)
    vmin_c, vmax_c = BOUNDS['cath']; vmin_g, vmax_g = BOUNDS['gate']

    rows = []
    prev_theta = None # memory between E targets (warm start)

    for E0_cm in E_targets_Vcm:
        E0 = float(E0_cm)*100.0  # convert V/cm → V/m

        # --- Define parameter bounds and seeds ---
        def theta_bounds():
            b = [(vmin_g, min(vmax_g, GATE_MAX_BY_ANODE))]
            if opt_pmts:
                b += [BOUNDS['pmt_top'], BOUNDS['pmt_bot']]
            if opt_rings:
                b += [(1e-4, 10.0)]*(nR+1)
            if opt_offset:
                b += [delta_bounds]
            return b
        bnds = theta_bounds()

        def make_theta(seed_vg):
            """Construct initial parameter vector θ."""
            th = [float(seed_vg)]
            if opt_pmts: th += [PMT_TOP_V, PMT_BOT_V]
            if opt_rings: th += [1.0]*(nR+1)
            if opt_offset: th += [0.0]
            return np.array(th, float)

        # Seeds for V_gate exploration
        seeds = []
        if prev_theta is not None:
            vg_center = float(prev_theta[0])
            seeds = list(np.linspace(vg_center - 300, vg_center + 300, 7))

        for vg0 in vg_seeds: 
            vg0 = np.clip(vg0, vmin_g, min(vmax_g, GATE_MAX_BY_ANODE))
            if (vg0 + EXTRACTION) <= ANODE_MAX:
                seeds.append(float(vg0))
 
        best_res, best_theta = None, None

        # --- Iterate over all gate seeds ---
        for vg0 in seeds:
            th0 = make_theta(vg0)
            if prev_theta is not None:
                k = min(len(th0), len(prev_theta))
                th0[:k] = 0.7*th0[:k] + 0.3*prev_theta[:k] # slight inertia

            def cost_theta(theta):
                """Main cost function for L-BFGS-B."""
                j = 0
                Vg = float(theta[j]); j += 1
                V = {}

                # Optional PMT optimization
                if opt_pmts:
                    V['pmt_top'] = float(theta[j]); V['pmt_bot'] = float(theta[j+1]); j += 2
                else:
                    V['pmt_top'] = PMT_TOP_V; V['pmt_bot'] = PMT_BOT_V

                # Ring configuration
                if opt_rings:
                    deltas = np.clip(theta[j:j+nR+1], 1e-4, None)
                    j += (nR + 1)
                elif (rings_top_factor is not None) and ring_names:
                    deltas = ring_weights_uniform_plus_top(ring_names, top_factor=rings_top_factor)
                elif (rings_power_alpha is not None) and ring_names:
                    deltas = ring_weights_from_power_alpha(ring_names, rings_power_alpha)
                else:
                    deltas = np.ones(nR + 1, float)

                # Solve for Vc matching target E0
                Vc, hit_bound, _ = solve_cathode_for_E0(
                    Bstack, Vg, EXTRACTION, M_LXE, E0,
                    V, ring_names, vmin_c, vmax_c, tol=50.0,
                    ring_deltas=deltas, drift_sign=-1,
                    enforce_order=True, eps_order=order_margin_V,
                    enforce_sign=True, eps_sign=25.0
                )

                names_bu, w_pos = segment_weights_from_positions(ring_names, RING_Z, z_cath=Z_CATH, z_gate=Z_GATE)
                w_use = np.asarray(deltas if deltas is not None else w_pos, float)
                V.update(rings_from_segment_weights(Vc, Vg, names_bu, w_use))

                V['gate']  = Vg
                V['cath']  = Vc
                V['anode'] = Vg + EXTRACTION

                # Optional global voltage offset
                delta = float(theta[j]) if opt_offset else 0.0
                if opt_offset:
                    V['gate']  += delta; V['cath'] += delta; V['anode'] += delta
                    for name in ring_names: V[name] += delta

                # High-voltage boundary penalties
                pen_lim = 0.0
                pen_lim += 0.0 if BOUNDS['gate'][0] <= V['gate'] <= BOUNDS['gate'][1] else 1.0
                pen_lim += 0.0 if BOUNDS['cath'][0] <= V['cath'] <= BOUNDS['cath'][1] else 1.0
                pen_lim += 0.0 if (V['anode'] <= ANODE_MAX) else 1.0

                # Compute field and metrics
                _, _, Er, Ez = superpose_fast(Bstack, V)
                Ed_med, dev_mean, inhEz = drift_metrics_rel(Er, Ez, M_LXE)
                theta_med = theta_med_deg_balanced(Er, Ez, M_LXE, z) 

                # Core term: average field deviation
                main = dev_mean
                rel_err = (Ed_med - E0) / max(E0, 300.0)
                pen_E0 = 200.0 * rel_err**2
                pen = pen_E0 + 1e3*pen_lim

                # Additional penalties
                revfr  = frac_reversed(Ez, M_BELOW, expected_sign=-1)

                return (main + lambda_inh*inhEz + pen + w_rev*revfr)

            # Run optimizer
            res = minimize(cost_theta, th0, method='L-BFGS-B',
                           bounds=bnds, options={'maxiter': 300})
            if (best_res is None) or (res.fun < best_res.fun):
                best_res, best_theta = res, res.x

        prev_theta = best_theta.copy()

        # --- Reconstruct final best configuration ---
        j = 0
        Vg_opt = float(best_theta[j]); j += 1
        Vopt = {}

        if opt_pmts:
            Vopt['pmt_top'] = float(best_theta[j]); Vopt['pmt_bot'] = float(best_theta[j+1]); j += 2
        else:
            Vopt['pmt_top'] = PMT_TOP_V; Vopt['pmt_bot'] = PMT_BOT_V

        if opt_rings:
            deltas_final = np.clip(best_theta[j:j+nR+1], 1e-4, None); j += (nR + 1)
        elif (rings_top_factor is not None) and ring_names:
            deltas_final = ring_weights_uniform_plus_top(ring_names, top_factor=rings_top_factor)
        elif (rings_power_alpha is not None) and ring_names:
            deltas_final = ring_weights_from_power_alpha(ring_names, rings_power_alpha)
        else:
            deltas_final = np.ones(nR + 1, float)

        Vc_opt, hit_bound, _ = solve_cathode_for_E0(
            Bstack, Vg_opt, EXTRACTION, M_LXE, E0,
            Vopt, ring_names, vmin_c, vmax_c, tol=50.0,
            ring_deltas= deltas_final, drift_sign=-1,
            enforce_order=True, eps_order=order_margin_V,
            enforce_sign=True, eps_sign=25.0
        )

        delta_opt = float(best_theta[j]) if opt_offset else 0.0

        names_bu, w_pos = segment_weights_from_positions(ring_names, RING_Z, z_cath=Z_CATH, z_gate=Z_GATE)
        w_use = np.asarray(deltas_final if deltas_final is not None else w_pos, float)
        Vopt.update(rings_from_segment_weights(Vc_opt, Vg_opt, names_bu, w_use))

        Vopt['gate']  = Vg_opt + (delta_opt if opt_offset else 0.0)
        Vopt['cath']  = Vc_opt + (delta_opt if opt_offset else 0.0)
        Vopt['anode'] = Vg_opt + EXTRACTION + (delta_opt if opt_offset else 0.0)

        # Final metrix
        _, _, Er, Ez = superpose_fast(Bstack, Vopt)
        Ed_med2, dev_mean2, inhEz2 = drift_metrics_rel(Er, Ez, M_LXE)
        theta_med = theta_med_deg_balanced(Er, Ez, M_LXE, z)
        drmean_mm = deflection_mean_mm(Er, Ez, r, z, M_LXE)

        rows.append(dict(
            Ecm=E0_cm,
            V_gate=float(Vopt['gate']),
            V_cath=float(Vopt['cath']),
            V_anode=float(Vopt['anode']),
            V_pmt_top=float(Vopt['pmt_top']),
            V_pmt_bot=float(Vopt['pmt_bot']),
            mean=float(dev_mean2),
            inhEz=float(inhEz2),
            Ed_kVcm=float(Ed_med2/1e5),
            hit_bound=int(hit_bound),
            theta_med_bal_deg=float(theta_med),
            deflect_mean_mm=float(drmean_mm),
            **{f"V_{name}": float(Vopt.get(name, np.nan)) for name in ring_names}
        ))

    return pd.DataFrame(rows)


def add_Egas_and_save(df, tag):
    """
    Compute gas amplification fields, append to DataFrame, and save results.

    For each optimized configuration:
      - Compute median extraction field in the gas region (kV/cm)
      - Save to CSV
      - Generate summary plots of voltages, deviation, deflection, etc.
    """

    Eg_list = []
    for _, row in df.iterrows():
        # Build voltage dictionary for current configuration
        V = dict(gate=row['V_gate'], cath=row['V_cath'], anode=row['V_anode'],
                 pmt_top=row['V_pmt_top'], pmt_bot=row['V_pmt_bot'])
        Eg_list.append(median_gas_field_from_V(Bstack, V, roi_gas=ROI_GXe))  # kV/cm

    df['Egas_kVcm'] = Eg_list
    df.to_csv(f"optimal_voltages_3D_{tag}.csv", index=False)

    # Generate standard diagnostic plots
    plot_optimal(df, out=f"optimal_voltages_vs_E_3D_{tag}.png")
    plot_deviation(df, out=f"deviation_vs_E_3D_{tag}_th.png")
    plot_deflection(df, out=f"deflection_vs_E_3D_{tag}.png")
    plot_deltaEz(df, out=f"deltaEz_vs_E_3D_{tag}.png")
    plot_theta(df,   out=f"theta_vs_E_3D_{tag}.png")


def plot_mean_multi(dfs, labels, out="deviation_vs_E_3D_compare_gate_2.png"):
    """
    Compare mean field deviations across multiple optimization series.
    """

    plt.figure(figsize=(6.2, 3.8))
    for df, lab in zip(dfs, labels):
        ycol = 'mean' if 'mean' in df.columns else None
        if ycol is None:
            print(f"[WARN] DF sin 'mean'/'mean' para {lab}; lo salto.")
            continue
        plt.semilogy(df['Ecm'], 100.0*df[ycol], label=lab)

    # Mark boundary hits (hit_bound == 1) if present
    for df, _ in zip(dfs, labels):
        if ('hit_bound' in df.columns) and (('mean' in df.columns) or ('mean' in df.columns)):
            m = df['hit_bound'] > 0
            ycol = 'mean' if 'mean' in df.columns else 'mean'
            if m.any():
                plt.scatter(df.loc[m, 'Ecm'], 100.0*df.loc[m, ycol], s=16, marker='x')

    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Deviation [%]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()



# ============================================================
# 11. PARAMETRIC SWEEPS AND FIELD DIAGNOSTICS
# ============================================================

def leak_mean_band(Er, Ez, mask):
    """
    Compute mean(|Er|/|Ez|) within a leakage band (below the gate).

    - Applies a numerical floor to Ez.
    - Used to quantify horizontal field leakage near the liquid–gas interface.
    """
    idx = np.where(mask)
    if idx[0].size == 0: return 0.0
    Erb = np.abs(Er[idx]); Ezb = np.abs(Ez[idx])
    floor = max(1.0, np.percentile(Ezb, 10))
    r = Erb / np.clip(Ezb, floor, None)
    r = r[np.isfinite(r)]
    return float(np.nanmean(r)) if r.size else 0.0


def theta_med_deg_balanced(Er, Ez, mask, z):
    """
    Compute median drift angle θ (deg) using z-median per column.
    This version is robust against outliers and low-field instabilities.
    """
    idx = np.where(mask)
    if idx[0].size == 0: return 0.0
    Er_sel = np.where(mask, Er, np.nan)
    Ez_sel = np.where(mask, Ez, np.nan)
    Ez_abs = np.abs(Ez_sel)
    floor  = np.nanmax([1.0, np.nanpercentile(Ez_abs, 10)])
    ratio  = np.abs(Er_sel) / np.clip(Ez_abs, floor, None)
    theta  = np.arctan(ratio)                      # rad
    theta_med_z = np.nanmedian(theta, axis=0)      # (Nr,)
    th_med = float(np.nanmedian(np.abs(theta_med_z)))
    return np.degrees(th_med)


def sweep_gate_for_E0(Bstack, E0_cm=60.0, roi_lxe=ROI_LXe,
                      gate_min=None, gate_max=None, n_steps=121,
                      leak_band_mm=(2.0, 0.2), out_csv="gate_sweep_E60.csv",
                      out_png="gate_sweep_E60.png", top_factor=None, power_alpha=None):
    """
    Sweep the gate voltage (V_gate) at a fixed target drift field (E0_cm).

    For each gate voltage:
      - The cathode voltage (V_cath) is solved via bisection to satisfy
        median(Ez) ≈ -E0.
      - The anode voltage is set by V_anode = V_gate + EXTRACTION.
      - The field map is reconstructed, and key metrics are computed:
        * mean(|Er|/|Ez|) → lateral leakage
        * inhEz → field inhomogeneity (RMS)
        * leak_mean → near-gate leakage
        * theta_med_deg → median drift angle

    The function stores the results in a CSV file and generates a diagnostic
    plot showing how field uniformity evolves with gate bias.
    """

    import numpy as np, pandas as pd, matplotlib.pyplot as plt

    r, z = Bstack['r'], Bstack['z']
    M_LXE = roi_mask(r, z, *roi_lxe)

    # Define wide band below the gate for leakage evaluation
    dz_hi, dz_lo = leak_band_mm[0]*1e-3, leak_band_mm[1]*1e-3
    M_LXE  = roi_mask(Bstack['r'], Bstack['z'], *ROI_LXe)  & SUPPORT
    ROI_TOP_WIDE = (ROI_R[0], ROI_R[1], d_gate - dz_hi, d_gate - dz_lo)
    M_TOP = roi_mask(r, z, *ROI_TOP_WIDE) & SUPPORT
    M_BELOW= roi_mask(r, z, *ROI_BELOW_CATH) & SUPPORT

    ring_names = list_rings_from_basis(Bstack); nR = len(ring_names)

    # Voltage range for the scan
    if gate_min is None: gate_min = BOUNDS['gate'][0]
    if gate_max is None: gate_max = min(BOUNDS['gate'][1], GATE_MAX_BY_ANODE)
    gate_grid = np.linspace(gate_min, gate_max, n_steps)

    rows = []
    E0 = float(E0_cm) * 100.0  # Convert V/cm → V/m

    for Vg in gate_grid:
        Va = Vg + EXTRACTION
        if Va > ANODE_MAX:
            continue  # respect anode limit

        # Ring weighting scheme
        deltas = None
        if (top_factor is not None) and (nR > 0):
            deltas = ring_weights_uniform_plus_top(ring_names, top_factor=top_factor)
        elif (power_alpha is not None) and (nR > 0):
            deltas = ring_weights_from_power_alpha(ring_names, power_alpha)
        else:
            deltas = np.ones(nR + 1, float)

        baseV = dict(pmt_top=PMT_TOP_V, pmt_bot=PMT_BOT_V)

        # Solve cathode voltage for target drift field
        Vc, hit, Ed_signed = solve_cathode_for_E0(
            Bstack, Vg, EXTRACTION, M_LXE, E0,
            baseV, ring_names, BOUNDS['cath'][0], BOUNDS['cath'][1],
            tol=50.0, ring_deltas=deltas, drift_sign=-1,
            enforce_order=True, eps_order=10.0,
            enforce_sign=True,  eps_sign=25.0
        )

        # Build complete voltage map
        V = {'gate': Vg, 'cath': Vc, 'anode': Va,
             'pmt_top': PMT_TOP_V, 'pmt_bot': PMT_BOT_V}
        names_bu, w_pos = segment_weights_from_positions(ring_names, RING_Z, z_cath=Z_CATH, z_gate=Z_GATE)
        w_use = np.asarray(deltas if deltas is not None else w_pos, float)
        V.update(rings_from_segment_weights(Vc, Vg, names_bu, w_use))

        # Compute field and derived metrics
        _, _, Er, Ez = superpose_fast(Bstack, V)
        Ed_med, dev_mean, inhEz = drift_metrics_rel(Er, Ez, M_LXE)
        leak_mean = leak_mean_band(Er, Ez, M_TOP)
        theta_med = theta_med_deg_balanced(Er, Ez, M_LXE, z)

        rows.append(dict(
            V_gate=Vg, V_cath=Vc, V_anode=Va,
            mean=dev_mean, inhEz=inhEz,
            score=dev_mean + 0.6*inhEz + 1.2*leak_mean,
            leak_mean=leak_mean, Ed_kVcm=Ed_med/1e5, hit_bound=int(hit),
            theta_med_deg=float(theta_med)
        ))


    df = pd.DataFrame(rows).sort_values("V_gate").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv} with {len(df)} points.")

    # ----------- Plotting -----------
    fig, ax = plt.subplots(figsize=(6.2,3.8))
    ax.semilogy(df['V_gate'], 100.0*df['mean'], label='mean: $E_r/E_z$ (LXe)')
    ax.semilogy(df['V_gate'], 100.0*df['inhEz'], label='inhomog $E_z$ (RMS)')
    ax.set_xlabel(r'$V_{\rm gate}$ [V]')
    ax.set_ylabel('Deviation [%]')
    ax.grid(True, which='both', ls=':')
    ax.legend(loc='best')

    # Mark points where the solver hits limits
    m = df['hit_bound'] > 0
    if m.any():
        ax.scatter(df.loc[m,'V_gate'], 100.0*df.loc[m,'mean'], s=18, c='k', marker='x', label='hit_bound')

    # Secondary axis: cathode voltage
    ax2 = ax.twinx()
    ax2.plot(df['V_gate'], df['V_cath']/1e3, ls='--')
    ax2.set_ylabel(r'$V_{\rm cath}$ [kV]')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_png}")

    return df


def theta_mean_deg(Er, Ez, mask):
    """
    Compute mean drift angle θ = arctan(|Er|/|Ez|) in degrees.
    """
    idx = np.where(mask)
    if idx[0].size == 0:
        return 0.0
    Er_sel = Er[idx]; Ez_sel = Ez[idx]
    finite = np.isfinite(Er_sel) & np.isfinite(Ez_sel)
    Er_sel = Er_sel[finite]; Ez_sel = Ez_sel[finite]
    if Er_sel.size == 0:
        return 0.0

    Er_abs = np.abs(Er_sel)
    Ez_abs = np.abs(Ez_sel)
    floor  = max(1.0, np.percentile(Ez_abs, 10))
    ratio  = Er_abs / np.clip(Ez_abs, floor, None)  # ~tan(theta)
    theta  = np.arctan(ratio)                       # rad
    th90   = float(np.quantile(np.abs(theta), 0.90))
    return np.degrees(th90)

def deflection_mean_mm(Er, Ez, r, z, mask):
    """
    Compute the mean drift deflection in mm within the ROI.

    Integrates |Er/Ez| along z (using trapezoidal rule) for each r column.
    Returns the 90th percentile of |Δr| in millimeters.
    """
    Nz, Nr = Ez.shape
    Ez_abs = np.abs(Ez)
    floor  = max(1.0, np.nanpercentile(Ez_abs[mask], 10)) if np.any(mask) else 1.0
    ratio  = np.abs(Er) / np.clip(Ez_abs, floor, None)  # ~tan(theta)

    z = np.asarray(z, float)
    dz_rows = np.empty(Nz)
    dz_rows[1:-1] = 0.5*(z[2:]-z[:-2])
    dz_rows[0]    = z[1]-z[0]
    dz_rows[-1]   = z[-1]-z[-2]

    dr_list = []
    for j in range(Nr):
        mj = mask[:, j]
        if np.count_nonzero(mj) < 2:
            continue
        rj = ratio[mj, j]
        dzj = dz_rows[mj]
        
        rj_mid = 0.5*(rj[:-1] + rj[1:])
        dz_mid = 0.5*(dzj[:-1] + dzj[1:])
        dr = np.nansum(rj_mid * dz_mid)  
        dr_list.append(abs(dr)*1000.0)   # convert to mm

    if not dr_list:
        return 0.0
    return float(np.nanquantile(dr_list, 0.90))

# ============================================================
# Plot utilities
# ============================================================

def plot_deltaEz(df, out='deltaEz_vs_E.png'):
    """
    Plot axial field inhomogeneity (inhEz) vs median drift field.
    """
    plt.figure(figsize=(5.8, 3.6))
    y = 100.0 * df['inhEz']
    plt.semilogy(df['Ecm'], y)
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Axial inhomogeneity [%]')
    plt.grid(True, which='both', ls=':')
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_theta(df, out='theta_vs_E.png'):
    """
    Plot drift angle θ vs drift field using balanced definition.
    """
    if 'theta_med_bal_deg' not in df.columns: return

    plt.figure(figsize=(5.8, 3.6))
    plt.plot(df['Ecm'], df['theta_med_bal_deg'], label=r'$\mathrm{median}(\theta)$ (deg)')
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Drift angle [deg]')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

def plot_deltaEz_multi(dfs, labels, out="deltaEz_vs_E_multi.png"):
    """
    Compare axial inhomogeneity across multiple configurations.
    """
    plt.figure(figsize=(6.2, 3.8))
    for df, lab in zip(dfs, labels):
        if 'inhEz' in df.columns:
            plt.semilogy(df['Ecm'], 100.0*df['inhEz'], label=lab)
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Axial inhomogeneity [%]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_theta_multi(dfs, labels, out="theta_vs_E_multi.png"):
    """
    Compare drift angle vs field across multiple configurations.
    """
    plt.figure(figsize=(6.2, 3.8))
    for df, lab in zip(dfs, labels):
        if 'theta_med_bal_deg' in df.columns:
            plt.plot(df['Ecm'], df['theta_med_bal_deg'], label=lab)
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Drift angle [deg]')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

# ============================================================
# Experiments: ring geometry tuning
# ============================================================

def experiment_top_ring(Bstack, E0_cm=60.0,
                        factors=(1.00, 1.05, 1.10, 1.12)):
    """
    Perform gate-voltage sweeps for several 'top_factor' values.
    Each factor modifies the last ring segment (near the gate).
    """
    import pandas as pd
    ring_names = list_rings_from_basis(Bstack)
    tagE = f"E{int(round(E0_cm))}"
    best_rows = []

    for f in factors:
        csv = f"gate_sweep_{tagE}_topF_{str(f).replace('.','p')}.csv"
        png = f"gate_sweep_{tagE}_topF_{str(f).replace('.','p')}.png"
        df = sweep_gate_for_E0(Bstack, E0_cm=E0_cm,
                               gate_min=max(-800.0, BOUNDS['gate'][0]),
                               gate_max=min(+1200.0, GATE_MAX_BY_ANODE),
                               n_steps=121,
                               out_csv=csv, out_png=png,
                               top_factor=f)
        if len(df):
            i = df['score'].idxmin()
            row = dict(df.loc[i])
            row['top_factor'] = f
            best_rows.append(row)

    out = pd.DataFrame(best_rows)
    out.to_csv(f"best_top_ring_{tagE}.csv", index=False)
    print(f"[OK] Saved best_top_ring_{tagE}.csv")
    return out

def experiment_power_alpha(Bstack, E0_cm=60.0,
                           alphas=(0.7, 0.9, 1.0, 1.2, 1.5, 2.0)):
    """
    Same as `experiment_top_ring`, but tests power-law ring spacing (t**alpha).
    """
    import pandas as pd
    tagE = f"E{int(round(E0_cm))}"
    best_rows = []

    for a in alphas:
        csv = f"gate_sweep_{tagE}_alpha_{str(a).replace('.','p')}.csv"
        png = f"gate_sweep_{tagE}_alpha_{str(a).replace('.','p')}.png"
        df = sweep_gate_for_E0(
            Bstack, E0_cm=E0_cm,
            gate_min=max(-800.0, BOUNDS['gate'][0]),
            gate_max=min(+1200.0, GATE_MAX_BY_ANODE),
            n_steps=121,
            out_csv=csv, out_png=png,
            power_alpha=a
        )
        if len(df):
            i = df['score'].idxmin()
            row = dict(df.loc[i])
            row['alpha'] = a
            best_rows.append(row)

    out = pd.DataFrame(best_rows)
    out.to_csv(f"best_power_alpha_{tagE}.csv", index=False)
    print(f"[OK] Saved best_power_alpha_{tagE}.csv")
    return out


def optimize_multi_alpha(Bstack, E_targets_Vcm, roi_lxe, alphas):
    """
    Run voltage optimization for multiple ring-spacing exponents (alpha).
    """
    dfs, labels = [], []
    for a in alphas:
        tag = f"alpha_{str(a).replace('.','p')}_7rings_roi"
        df = optimize_voltages_constrained(
            Bstack, E_targets_Vcm, roi_lxe=roi_lxe,
            opt_pmts=False, opt_rings=False,
            rings_power_alpha=a,
        )
        add_Egas_and_save(df, tag)
        dfs.append(df); labels.append(f"α={a}")

    # Comparison plots across alpha values
    plot_mean_multi(dfs, labels, out="deviation_vs_E_3D_MULTI_poweralpha.png")
    plot_deltaEz_multi(dfs, labels, out="deltaEz_vs_E_3D_MULTI_poweralpha.png")
    plot_theta_multi(dfs, labels, out="theta_vs_E_3D_MULTI_poweralpha.png")
    return dfs


def plot_deltaE_map(Bstack, Vdict, roi=None, out="deltaE_map.png", signed=False):
    """
    Visualize axial field inhomogeneity ΔE_z / E_z,median on the (r²,z) plane.
    """
    r, z, Er, Ez = superpose_fast(Bstack, Vdict)
    M = SUPPORT.copy()
    if roi is not None:
        M &= roi_mask(r, z, *roi)

    Ez_abs = np.abs(Ez)
    Ed_med = float(np.nanmedian(Ez_abs[M]))
    eps = max(1e-12, 1e-3*Ed_med)  # numerical cushion

    if signed:
        # Signed deviation (positive = above median)
        delta = (np.where(M, Ez_abs, np.nan) - Ed_med) / (Ed_med + eps)
        vmin, vmax = -0.1, 0.1
        cbar_label = r'$(|E_z|-\mathrm{med}(|E_z|))/\mathrm{med}(|E_z|)$'
    else:
        # Absolute deviation (magnitude of inhomogeneity)
        delta = np.abs(np.where(M, Ez_abs, np.nan) - Ed_med) / (Ed_med + eps)
        vmin, vmax = 0.0, 0.1
        cbar_label = r'$|\,|E_z|-\mathrm{med}(|E_z|)\,|/\mathrm{med}(|E_z|)$'

    # Restrict to ROI for display
    rows = np.where(np.any(M, axis=1))[0]; cols = np.where(np.any(M, axis=0))[0]
    z_show, r_show = z[rows], r[cols]; Dshow = delta[np.ix_(rows, cols)]

    R2 = (np.meshgrid(r_show, z_show)[0]**2)*1e4
    extent = [np.nanmin(R2), np.nanmax(R2), np.nanmin(z_show)*100, np.nanmax(z_show)*100]

    plt.figure(figsize=(6,3.8))
    im = plt.imshow(Dshow, origin='lower', extent=extent, aspect='auto', vmin=vmin, vmax=vmax)
    im.cmap.set_bad(im.cmap(0.0), 1.0)
    cbar = plt.colorbar(im); cbar.set_label(cbar_label)
    plt.xlabel(r'$r^2$ (cm$^2$)'); plt.ylabel('z (cm)')
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()


def _dbg_roi_count(mask, name="ROI"):
    """
    Debug utility: print number of points within a ROI mask.

    Example
    -------
    >>> _dbg_roi_count(M_LXE, name="LXe Region")
    [DBG] LXe Region: 15023/24000 points
    """
    n = int(np.count_nonzero(mask))
    N = int(mask.size)
    print(f"[DBG] {name}: {n}/{N} points")


# ============================================================
#  MAIN EXECUTION SCRIPT
#  Field optimization and diagnostics for LXe TPC geometry
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # 0) Load precomputed electrode basis and cached metadata
    # --------------------------------------------------------
    Bstack = load_basis_cache(PATH_BASIS, Nr=NR_BINS, rmax=RMAX_FOR_BIN)
    r, z = Bstack['r'], Bstack['z']
    SUPPORT = basis_support_mask(Bstack)
    print(">>> Loaded electrode basis:", Bstack['names'])

    # --- Quick check of basis metadata cache ---
    try:
        npz = np.load(CACHE_NPZ, allow_pickle=True, mmap_mode='r')
        meta = npz['meta'].item()
        print("Cache metadata:", {k: meta[k] for k in ['version','path','Nr','rmax','size']})
    except Exception as e:
        print("[WARN] Cache metadata not found or invalid:", e)

    # --------------------------------------------------------
    # 1) Optional validation against direct COMSOL or CSV data
    # --------------------------------------------------------
    if DO_VALIDATION and os.path.exists(PATH_EXAMPLES):
        try:
            direct = load_direct_on_basis_grid_interp(PATH_EXAMPLES, Bstack, rmax=RMAX_FOR_BIN)
            Vref = Vdict_for_E(gate_voltage=0.0, pmt_top=PMT_TOP_V, pmt_bottom=PMT_BOT_V,
                               include_rings=True, basis_like=Bstack, ring_positions=RING_Z)
            plot_diff_map(Bstack, Vref, direct, full_extent=True, out='superpos_3D_check.png')
            print(">>> Validation plot saved: superpos_3D_check.png")
        except Exception as e:
            print(f"[WARNING] Validation skipped: {e}")

    # --------------------------------------------------------
    # 2) Define target drift-field range
    # --------------------------------------------------------
    targets = np.linspace(20.0, 600.0, 131)  # V/cm
    # For quick debugging, you can replace with e.g. np.arange(0, 1301, 100)

    # --------------------------------------------------------
    # 2b) ROI verification (apply support mask)
    # --------------------------------------------------------
    M_LXE = roi_mask(r, z, *ROI_LXe) & SUPPORT
    _dbg_roi_count(M_LXE, "ROI_LXe")

    print("Initialization complete")

    # ========================================================
    #  SCENARIO A — Fixed PMTs and fixed rings (linear ramp)
    # ========================================================
    df_base = optimize_voltages_constrained(
        Bstack, targets, roi_lxe=ROI_LXe,
        opt_pmts=False, opt_rings=False,
        lambda_inh=0.6, lambda_smooth=0.02,
        opt_offset=False
    )
    add_Egas_and_save(df_base, "base_fixed")

    # --- Field inhomogeneity map at 60 V/cm ---
    row60 = df_base.iloc[(df_base['Ecm'] - 60).abs().argmin()]
    V60   = dict(gate=row60['V_gate'], cath=row60['V_cath'], anode=row60['V_anode'],
                 pmt_top=row60['V_pmt_top'], pmt_bot=row60['V_pmt_bot'])
    plot_deltaE_map(Bstack, V60, roi=ROI_LXe, out="deltaE_map_E60_base.png")

    # ========================================================
    #  SCENARIO B — Free PMT voltages, fixed rings
    # ========================================================
    df_pmts = optimize_voltages_constrained(
        Bstack, targets, roi_lxe=ROI_LXe,
        opt_pmts=True, opt_rings=False,
        lambda_inh=0.6, lambda_smooth=0.02,
        opt_offset=False
    )
    add_Egas_and_save(df_pmts, "pmts_free")

    # ========================================================
    #  SCENARIO C — Modify only the last ring (top segment)
    # ========================================================
    last_ring_factors = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]  # 1.0=control
    dfs_last, labels_last = [], []
    for f in last_ring_factors:
        tag = f"lastRing_x{str(f).replace('.','p')}"
        df = optimize_voltages_constrained(
            Bstack, targets, roi_lxe=ROI_LXe,
            opt_pmts=False, opt_rings=False,
            rings_top_factor=f,    # <-- affects only the last inter-ring step
            lambda_inh=0.6, lambda_smooth=0.02,
            opt_offset=False
        )
        add_Egas_and_save(df, tag)
        dfs_last.append(df); labels_last.append(f"last ring ×{f}")

    # ========================================================
    #  SCENARIO D — Power-law ring weighting (t**alpha)
    # ========================================================
    alphas = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
    dfs_alpha, labels_alpha = [], []

    for a in alphas:
        tag = f"allRings_alpha{str(a).replace('.','p')}"
        df = optimize_voltages_constrained(
            Bstack, targets, roi_lxe=ROI_LXe,
            opt_pmts=False, opt_rings=False,
            rings_power_alpha=a,   # <-- modifies all ring weights
            lambda_inh=0.6, lambda_smooth=0.02,
            opt_offset=False
        )
        add_Egas_and_save(df, tag)
        dfs_alpha.append(df); labels_alpha.append(f"all rings α={a}")

    # ========================================================
    #  COMPARATIVE PLOTS
    # ========================================================

    # Base vs free-PMT comparison
    plot_mean_multi([df_base, df_pmts], ["base (PMTs OFF)", "PMTs ON"],
                   out="deviation_vs_E_base_vs_pmts.png")
    plot_deltaEz_multi([df_base, df_pmts], ["base", "PMTs ON"],
                       out="deltaEz_vs_E_base_vs_pmts.png")
    plot_theta_multi([df_base, df_pmts], ["base", "PMTs ON"],
                     out="theta_vs_E_base_vs_pmts.png")

    # Last-ring variation series
    plot_mean_multi(dfs_last, labels_last, out="deviation_vs_E_LASTRING.png")
    plot_deltaEz_multi(dfs_last, labels_last, out="deltaEz_vs_E_LASTRING.png")
    plot_theta_multi(dfs_last, labels_last, out="theta_vs_E_LASTRING.png")

    # Power-law α series
    plot_mean_multi(dfs_alpha, labels_last, out="deviation_vs_E_L.png")
    plot_deltaEz_multi(dfs_alpha, labels_last, out="deltaEz_vs_E_L.png")
    plot_theta_multi(dfs_alpha, labels_last, out="theta_vs_E_L.png")

    print(">>> DONE")