# =====================================================
# superposition_plots.py
# Author: Elena Muñoz
#
# This script loads unit field maps (1 V applied to each electrode),
# defines the detector geometry and regions of interest (ROI),
# and provides helper functions to read data and process the
# electric field in cylindrical coordinates (r, z, Er, Ez).
#
# The goal is to evaluate the uniformity of the electric field
# in the LXe region at low fields and to determine the optimal
# electrode voltage configuration.
# =====================================================

import numpy as np
import pandas as pd
import re, os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import minimize


# ============== CONFIGURATION ==============

# Paths to input CSV files
PATH_BASIS     = "/Volumes/NUEVO VOL/superposition_3D_27mm_27mm.csv"   # unit field maps (1 V applied)
PATH_EXAMPLES  = "example_3D_4000.csv"    # optional direct case for validation
DO_VALIDATION  = True                    # whether to compare with PATH_EXAMPLES

# --- Detector geometry (meters) ---
D_CG           = 30.8e-3                  # cathode↔gate distance (m) for analytical seed
d_cathode      = -0.0289/2 - 0.002       # cathode z-position (m)
d_gate         = +0.0289/2                # gate z-position (m)
d_anode        = +0.0289/2 + 0.0041       # anode z-position (m)

# --- ROI definitions (meters) ---
R_DET_M        = (31.0/2.0) * 1e-3
#ROI_R          = (0, (31.0/2.0) * 1e-3)
ROI_R          = (0, 10.0 * 1e-3)
MARGIN_Z       = 2e-3                     # 2 mm margin from electrodes

# LXe region (from cathode to gate)
ROI_LXe        = (ROI_R[0], ROI_R[1], d_cathode + MARGIN_Z, d_gate - MARGIN_Z)

# GXe region (from gate to anode)
ROI_GXe        = (ROI_R[0], ROI_R[1], d_gate + MARGIN_Z, d_anode - 5e-4)

# --- ROIs para penalizar fugas y hotspot (en metros) ---
# Banda inmediatamente por debajo del cátodo (captura hotspot y campo reverso con PMT inferior)
ROI_BELOW_CATH = (ROI_R[0], ROI_R[1], d_cathode - 6e-3, d_cathode - 1e-3)

# Banda en el tope del LXe, justo bajo el gate (captura la "fuga" desde la región de extracción)
ROI_TOP_LXE    = (ROI_R[0], ROI_R[1], d_gate - 1.0e-3,  d_gate - 3.0e-4)


# --- Fixed hardware voltages ---
EXTRACTION = 4000.0                       # Gate–Anode potential difference (V)
PMT_TOP_V  = -1000.0
PMT_BOT_V  = -1000.0

# --- HV limits ---
ANODE_MAX = +5000.0                       # maximum anode voltage (V)
GATE_MAX_BY_ANODE = ANODE_MAX - EXTRACTION

# Bounds for optimization (V)
BOUNDS = dict(
    gate=(-GATE_MAX_BY_ANODE, GATE_MAX_BY_ANODE),
    cath=(-5000.0, 2000.0),
    pmt_top=(-1100.0, -900.0),
    pmt_bot=(-1100.0, -900.0)
)

# Mapping from index (in the basis file) to electrode name
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

Z_CATH = -15.95e-3
Z_GATE = +14.05e-3
RING_Z = {
    'ring_7': -12.45e-3,
    'ring_6': -8.45e-3,
    'ring_5': -4.45e-3,
    'ring_4': -0.45e-3,
    'ring_3':  +3.55e-3,
    'ring_2':  +7.55e-3,
    'ring_1': +11.55e-3,   # near gate
}



# Radial binning settings
NR_BINS = 100
RMAX_FOR_BIN = ROI_R[1]

def _auto_cache_name(path_csv, Nr, rmax):
    base = os.path.splitext(os.path.basename(path_csv))[0]
    rmm = int(round(rmax * 1e3))
    return f"basis_cache_{base}_Nr{Nr}_R{rmm}mm.npz"

CACHE_NPZ = _auto_cache_name(PATH_BASIS, NR_BINS, RMAX_FOR_BIN)


# ============== Helper functions ==============
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
                names = [t.strip().lstrip('%').strip() for t in s.split(',')]
                break
    if names:
        return pd.read_csv(path, comment='%', header=None, names=names, engine='python')
    else:
        return pd.read_csv(path, comment='%', engine='python')

def _pick(df, *cands):
    """
    Returns the first column in `df` whose name matches any candidate in `cands`.
    """
    for c in cands:
        if c in df.columns:
            return c
    return None

def _pick_field_vcomp(df, which='Ex'):
    """
    Finds a column containing a specific electric field component (Ex, Ey, Ez),
    even if it has a prefix (e.g., 'es2.Ex (V/m)').
    """
    tail = f"{which} (V/m)"
    for col in df.columns:
        if str(col).strip().endswith(tail):
            return col
    pat = re.compile(rf"(^|[.\s_]){which}($|[\s(])", re.IGNORECASE)
    for col in df.columns:
        if pat.search(str(col)):
            return col
    return None

def _to_m_if_mm(arr_like):
    """
    Converts from mm to m if typical values are > 0.5 (assuming cm scale).
    Leaves values unchanged if they are already in meters.
    """
    a = np.asarray(arr_like, dtype=float)
    if np.nanpercentile(np.abs(a), 90) > 0.5:
        return a / 1000.0
    return a

def _unit_phi(x, y):
    """
    Given x and y, returns (cos(phi), sin(phi), phi) for the azimuthal angle phi.
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
    
    Parameters:
        df_slice : DataFrame with columns ['r','Er','Ez'] at fixed z
        Nr       : number of radial bins
        rmax     : maximum radius (m)
    
    Returns:
        r_centers : radial bin centers (m)
        Er_b      : averaged Er per bin
        Ez_b      : averaged Ez per bin
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
    vals = Ez[mask]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else 0.0


# ============== Basis loader (cartesian 3D → cylindrical r,z grids) ==============
def load_basis_cartesian(path, Nr=NR_BINS, rmax=RMAX_FOR_BIN):
    """
    Loads unit-field CSV (1 V per electrode), converts to cylindrical (r, Er, Ez),
    and aggregates onto a common r-grid (uniform in r^2) for each z-level and electrode.

    Returns:
        basis: dict mapping electrode -> {r, z, Er(Nz,Nr), Ez(Nz,Nr)}
               where r and z are shared across electrodes.
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

from scipy.interpolate import griddata

def load_direct_on_basis_grid_interp(path, Bstack, rmax=RMAX_FOR_BIN):
    """
    Loads a direct COMSOL CSV (all electrodes biased simultaneously),
    converts to cylindrical, and interpolates (Er, Ez) onto the basis (r,z) grid.

    Strategy:
      - Use linear interpolation where available.
      - Fill remaining NaNs with nearest-neighbour.
      - Report coverage diagnostics.

    Returns:
        dict(r, z, Er, Ez) aligned to Bstack['r'], Bstack['z'].
    """
    df = read_with_commented_header(path).rename(columns=lambda s: str(s).strip())

    xcol = _pick(df, 'x (mm)', 'x')
    ycol = _pick(df, 'y (mm)', 'y')
    zcol = _pick(df, 'z (mm)', 'z')
    Ex   = _pick_field_vcomp(df, 'Ex')
    Ey   = _pick_field_vcomp(df, 'Ey')
    Ez   = _pick_field_vcomp(df, 'Ez')
    assert all([xcol, ycol, zcol, Ex, Ey, Ez]), "Missing columns in direct CSV."

    # data (mm→m if needed)
    x = _to_m_if_mm(df[xcol].astype(float).values)
    y = _to_m_if_mm(df[ycol].astype(float).values)
    z = _to_m_if_mm(df[zcol].astype(float).values)
    ex = df[Ex].astype(float).values
    ey = df[Ey].astype(float).values
    ez = df[Ez].astype(float).values

    # cylindrical
    phi = np.arctan2(y, x); c = np.cos(phi); s = np.sin(phi)
    r  = np.sqrt(x*x + y*y)
    er = ex*c + ey*s

    # filter domain
    m = np.isfinite(r) & np.isfinite(z) & np.isfinite(er) & np.isfinite(ez) & (r <= rmax)
    r, z, er, ez = r[m], z[m], er[m], ez[m]

    # target grid from basis
    r_grid = np.asarray(Bstack['r'], dtype=float)
    z_grid = np.asarray(Bstack['z'], dtype=float)
    Rg, Zg = np.meshgrid(r_grid, z_grid)  # (Nz, Nr)

    pts = np.column_stack([r, z])

    # linear interpolation (inside convex hull)
    Er_lin = griddata(pts, er, (Rg, Zg), method='linear')
    Ez_lin = griddata(pts, ez, (Rg, Zg), method='linear')

    # nearest fill for holes (outside hull / sparse regions)
    Er_near = griddata(pts, er, (Rg, Zg), method='nearest')
    Ez_near = griddata(pts, ez, (Rg, Zg), method='nearest')

    Er = np.where(np.isfinite(Er_lin), Er_lin, Er_near).astype(np.float32)
    Ez = np.where(np.isfinite(Ez_lin), Ez_lin, Ez_near).astype(np.float32)

    # simple diagnostics
    lin_cov = np.isfinite(Er_lin).mean()
    print(f"[INFO] Direct→basis interpolation: linear coverage {100*lin_cov:.1f}% (rest filled with nearest).")

    return dict(r=r_grid.astype(np.float32), z=z_grid.astype(np.float32), Er=Er, Ez=Ez)

# ============== Direct loader (cartesian 3D → cylindrical, rebinned) ==============
def load_direct_cartesian(path, Nr=NR_BINS, rmax=RMAX_FOR_BIN):
    """
    Loads a direct COMSOL CSV (all electrodes biased), converts to cylindrical,
    and bins onto a common r-grid (uniform in r^2) at each available z-level.

    Note:
        z-levels must match (or be compatible with) the basis if you plan to
        compare element-wise. Otherwise, prefer `load_direct_on_basis_grid_interp`.
    """
    df = read_with_commented_header(path).rename(columns=lambda s: str(s).strip())

    xcol = _pick(df, 'x (mm)', 'x')
    ycol = _pick(df, 'y (mm)', 'y')
    zcol = _pick(df, 'z (mm)', 'z')
    Ex   = _pick_field_vcomp(df, 'Ex')
    Ey   = _pick_field_vcomp(df, 'Ey')
    Ez   = _pick_field_vcomp(df, 'Ez')
    assert all([xcol, ycol, zcol, Ex, Ey, Ez]), "Missing columns in direct CSV."

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

    # cylindrical
    c, s, _ = _unit_phi(df['x'].values, df['y'].values)
    df['r']  = np.sqrt(df['x'].values**2 + df['y'].values**2)
    df['Er'] = df['Ex'].values*c + df['Ey'].values*s

    # grids
    z_levels = np.sort(df['z'].unique())
    r_edges = _edges_uniform_r2(rmax, Nr)
    r_grid  = 0.5 * (r_edges[:-1] + r_edges[1:])

    Er_mat = np.zeros((len(z_levels), Nr), dtype=np.float32)
    Ez_mat = np.zeros_like(Er_mat)

    for i, z0 in enumerate(z_levels):
        sli = df[df['z'] == z0][['r', 'Er', 'Ez']]
        sli = sli[sli['r'] <= rmax]
        if sli.empty:
            Er_mat[i, :] = 0.0
            Ez_mat[i, :] = 0.0
            continue
        _, Er_b, Ez_b = _bin_to_r_grid(sli, Nr, rmax)
        Er_mat[i, :] = Er_b.astype(np.float32)
        Ez_mat[i, :] = Ez_b.astype(np.float32)

    return dict(r=r_grid.astype(np.float32), z=z_levels.astype(np.float32), Er=Er_mat, Ez=Ez_mat)


# ================= Cache and Fast Field Superposition =================

CACHE_VERSION = 2  # súbelo si cambias el formato

def build_basis_cache(path_csv, Nr=NR_BINS, rmax=RMAX_FOR_BIN, cache=CACHE_NPZ):
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
    # np.savez no guarda dicts directamente; lo metemos como objeto
    np.savez_compressed(cache,
                        names=np.array(names),
                        r=r, z=z, Er=Er, Ez=Ez,
                        meta=np.array([meta], dtype=object))
    return dict(names=names, r=r, z=z, Er=Er, Ez=Ez)

def load_basis_cache(path_csv, Nr=NR_BINS, rmax=RMAX_FOR_BIN, cache=CACHE_NPZ, force_rebuild=False):
    def _ok_meta(npz):
        if 'meta' not in npz.files:
            return False
        meta = npz['meta'].item()
        if meta.get('version') != CACHE_VERSION:
            return False
        # coherencia básica
        if int(meta.get('Nr', -1)) != int(Nr):
            return False
        if abs(float(meta.get('rmax', -1.0)) - float(rmax)) > 1e-9:
            return False
        # coherencia con el CSV origen (mtime/size)
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
        if _ok_meta(npz):
            return dict(names=list(npz['names']), r=npz['r'], z=npz['z'],
                        Er=npz['Er'], Ez=npz['Ez'])
        else:
            try:
                os.remove(cache)
            except Exception:
                pass  # si no podemos borrar, lo sobrescribimos luego

    # (re)construye
    return build_basis_cache(path_csv, Nr=Nr, rmax=rmax, cache=cache)

def superpose_fast(Bstack, Vdict):
    """
    Superpose electrode basis fields with given electrode voltages.

    Parameters:
        Bstack (dict): Contains 'names', 'r', 'z', 'Er', 'Ez'.
        Vdict (dict): Electrode voltages {electrode_name: voltage_value}.

    Returns:
        tuple: (r, z, Er_total, Ez_total)
    """
    names = Bstack['names']
    weights = np.array([Vdict.get(n, 0.0) for n in names], dtype=np.float32)  # (Ne,)
    Er_total = np.tensordot(weights, Bstack['Er'], axes=1)  # (Nz, Nr)
    Ez_total = np.tensordot(weights, Bstack['Ez'], axes=1)  # (Nz, Nr)
    return Bstack['r'], Bstack['z'], Er_total, Ez_total

# ================= ROI, Field Metrics, Voltage Setup =================
def roi_mask(r, z, rmin, rmax, zmin, zmax):
    """
    Create a boolean mask for points inside a cylindrical ROI.
    """
    R, Z = np.meshgrid(r, z)
    return (R >= rmin) & (R <= rmax) & (Z >= zmin) & (Z <= zmax)


def list_rings_from_basis(basis_like):
    """
    Return ordered list of ring electrode names from basis.
    """
    if isinstance(basis_like, dict) and 'names' in basis_like:
        names = basis_like['names']
    else:
        names = list(basis_like.keys())
    rings = sorted([k for k in names if str(k).startswith('ring_')], key=lambda s: int(str(s).split('_')[1]))
    return rings

def Vdict_for_E(gate_voltage=0.0, pmt_top=None, pmt_bottom=None,
                include_rings=True, basis_like=None, ring_positions=None):
    cathode_voltage = -4000.0
    anode_voltage   = gate_voltage + EXTRACTION
    V = dict(gate=gate_voltage, cath=cathode_voltage, anode=anode_voltage)

    if pmt_top is not None:
        V['pmt_top'] = pmt_top
    if pmt_bottom is not None:
        V['pmt_bot'] = pmt_bottom

    if include_rings and basis_like is not None:
        ring_names = list_rings_from_basis(basis_like)
        if ring_positions is not None:
            # Ladder por posición real (recomendado)
            V.update(rings_from_positions(cathode_voltage, gate_voltage,
                                          ring_names, ring_positions,
                                          z_cath=Z_CATH, z_gate=Z_GATE))
        else:
            # Fallback: rampa uniforme por índice (sólo si spacing ideal)
            n = len(ring_names)
            step = (gate_voltage - cathode_voltage) / (n + 1) if n > 0 else 0.0
            for i, name in enumerate(ring_names, start=1):
                # ring_1 es el más cercano al gate
                i_rev = (n + 1) - i
                V[name] = cathode_voltage + i_rev * step
    return V

def rings_from_positions(Vcath, Vgate, ring_names, ring_z, z_cath=Z_CATH, z_gate=Z_GATE):
    L = float(z_gate - z_cath)
    out = {}
    # Asegura orden claro: de abajo (cerca cátodo) a arriba (cerca gate)
    names_sorted = sorted(ring_names, key=lambda n: ring_z[n])
    for name in names_sorted:
        zi = float(ring_z[name])
        t = (zi - z_cath) / L  # 0 en cátodo, 1 en gate
        out[name] = Vcath + t * (Vgate - Vcath)
    return out


def segment_weights_from_positions(ring_names, ring_z, z_cath=Z_CATH, z_gate=Z_GATE):
    """
    Devuelve:
      - names_bottom_up: anillos de abajo (cerca cátodo) a arriba (cerca gate)
      - w: array de longitudes de los (n+1) tramos: [cath→r7, r7→r6, ..., r1→gate]
    """
    names_bottom_up = sorted(ring_names, key=lambda n: ring_z[n])  # z creciente
    zs = [z_cath] + [ring_z[n] for n in names_bottom_up] + [z_gate]
    w = np.diff(zs)  # (n+1,)
    return names_bottom_up, np.asarray(w, float)

def rings_from_segment_weights(Vc, Vg, names_bottom_up, w):
    w = np.clip(np.asarray(w, float), 1e-12, None)
    t_nodes = np.cumsum(w) / np.sum(w)            # fracción 0..1 en cada nodo
    t_rings = t_nodes[:-1]                        # excluye el nodo 'gate'
    out = {}
    for name, t in zip(names_bottom_up, t_rings):
        out[name] = Vc + t * (Vg - Vc)
    return out


def ring_weights_uniform_plus_top(ring_names, top_factor=1.0):
    """
    Devuelve el vector de pesos (n_rings+1) para rings_from_positions
    dejando TODOS los tramos uniformes (=1) salvo el penúltimo
    (entre ring_2 -> ring_1), que se multiplica por 'top_factor'.

    Notas:
      - ring_1 es el más cercano al gate (como en tu código).
      - El último peso (hueco ring_1 -> gate) se deja en 1.0 para que
        sólo cambie la posición de ring_1 respecto a la rampa uniforme.
    """
    n = len(ring_names)
    if n == 0:
        return np.array([1.0], float)

    w = np.ones(n + 1, dtype=float)   # n tramos entre anillos + 1 tramo hasta gate
    w[n - 1] = max(1e-4, float(top_factor))  # tramo ring_2 -> ring_1
    return w




def median_gas_field_from_V(Bstack, V, roi_gas):
    """
    Compute median electric field in the gas region for a given voltage setup.
    Returns field in kV/cm.
    """
    r, z, Er, Ez = superpose_fast(Bstack, V)
    Mgas = roi_mask(r, z, *roi_gas)
    Ez_abs = np.abs(Ez[Mgas])
    return (np.nanmedian(Ez_abs) / 1e5) if Ez_abs.size else np.nan


def p90_absE(Er, Ez, mask):
    """P90 de |E| en una banda (detecta hotspots)."""
    idx = np.where(mask)
    if idx[0].size == 0: return 0.0
    E = np.hypot(Er[idx], Ez[idx])
    E = E[np.isfinite(E)]
    return float(np.quantile(E, 0.90)) if E.size else 0.0

def frac_reversed(Ez, mask, expected_sign=-1):
    """Fracción de puntos con signo de Ez contrario al esperado (campo 'reverso')."""
    idx = np.where(mask)
    if idx[0].size == 0: return 0.0
    vals = Ez[idx]; vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0
    return float(np.mean(np.sign(vals) != expected_sign))

def p90_ratio_band(Er, Ez, mask):
    """P90 de Er/Ez en una banda (mide 'fuga' transversal)."""
    idx = np.where(mask)
    if idx[0].size == 0: return 0.0
    Erb = np.abs(Er[idx]); Ezb = np.abs(Ez[idx])
    if Erb.size == 0: return 0.0
    floor = max(1.0, np.percentile(Ezb, 10))
    r = Erb / np.clip(Ezb, floor, None)
    r = r[np.isfinite(r)]
    return float(np.quantile(r, 0.90)) if r.size else 0.0




# ========================= PLOTS =========================

def plot_diff_map(Bstack, Vdict, direct, roi_lxe=None, full_extent=False,
                  r2_max_cm2=None, out='diff_map_8.png'):
    """
    Plot |E_sup - E_dir| / |E_dir| on the (r^2, z) plane.

    Args:
        Bstack: basis stack dict used by superpose_fast.
        Vdict:  voltage map for the superposition.
        direct: dict(r, z, Er, Ez) on the SAME grid as Bstack (or already interpolated).
        roi_lxe: (rmin, rmax, zmin, zmax) to optionally crop and draw dashed lines.
        full_extent: if True, ignore roi crop (still can draw dashed lines if roi_lxe provided).
        r2_max_cm2: if set, additionally crop r by sqrt(r2_max_cm2)/100 m.
        out: output PNG filename.
    """
    r, z, Er, Ez = superpose_fast(Bstack, Vdict)
    Er_d, Ez_d   = direct['Er'], direct['Ez']

    num = np.sqrt((Er - Er_d)**2 + (Ez - Ez_d)**2)
    den = np.sqrt(Er_d**2 + Ez_d**2)

    valid = np.isfinite(num) & np.isfinite(den)
    den_med = np.nanmedian(den[valid & (den > 0)])
    eps = max(1e-12, 1e-3*(den_med if np.isfinite(den_med) else 1.0))

    Q = np.full_like(den, np.nan, dtype=float)
    keep = valid & (den > eps)
    Q[keep] = num[keep] / den[keep]

    # Optional crop to ROI just for display (and axes limits)
    r_show, z_show, Q_show = r, z, Q
    if (roi_lxe is not None) and (not full_extent):
        rmin, rmax, zmin, zmax = roi_lxe
        R, Z = np.meshgrid(r, z)
        M = (R>=rmin)&(R<=rmax)&(Z>=zmin)&(Z<=zmax)
        rows = np.where(np.any(M, axis=1))[0]
        cols = np.where(np.any(M, axis=0))[0]
        z_show = z[rows]; r_show = r[cols]; Q_show = Q[np.ix_(rows, cols)]

    # Optional crop by r^2 max (in cm^2)
    if r2_max_cm2 is not None:
        r_cut = (r2_max_cm2**0.5)/100.0
        keep_r = r_show <= r_cut
        r_show = r_show[keep_r]
        Q_show = Q_show[:, keep_r]

    # Build extent in (r^2[cm^2], z[cm])
    R2 = (np.meshgrid(r_show, z_show)[0]**2) * 1e4
    extent = [np.nanmin(R2), np.nanmax(R2), np.nanmin(z_show)*100, np.nanmax(z_show)*100]

    print("rel.err stats: min=", np.nanmin(Q_show), "med=", np.nanmedian(Q_show), "max=", np.nanmax(Q_show))

    plt.figure(figsize=(6, 3.8))
    im = plt.imshow(Q_show, origin='lower', extent=extent, aspect='auto',
                    norm=LogNorm(vmin=1e-4, vmax=2e-1))
    im.cmap.set_bad(im.cmap(0.0), 1.0)

    # Draw dashed lines at LXe ROI if provided
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


def plot_optimal(df, out='optimal_voltages_vs_E_3D_2_8.png'):
    """
    Plot optimal (Vcath, Vgate, Vanode) vs target median drift field,
    along with median amplification field in GXe if provided (Egas_kVcm).
    """
    fig, ax = plt.subplots(figsize=(5.9, 3.9))
    ax.plot(df['Ecm'], df['V_cath']/1e3, label=r'$U_{\mathrm{cathode}}$')
    ax.plot(df['Ecm'], df['V_gate']/1e3,  label=r'$U_{\mathrm{gate}}$')
    ax.plot(df['Ecm'], df['V_anode']/1e3, label=r'$U_{\mathrm{anode}}$')
    ax.set_xlabel('Median drift field [V/cm]')
    ax.set_ylabel('Electrode voltage [kV]')

    if 'Egas_kVcm' in df.columns and np.any(np.isfinite(df['Egas_kVcm'])):
        ax2 = ax.twinx()
        ax2.plot(df['Ecm'], df['Egas_kVcm'], ls='--')
        ax2.set_ylabel('Median amplification field [kV/cm]')

    fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.98))
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_deviation(df, out='deviation_vs_E_3D_2_8.png'):
    plt.figure(figsize=(5.8, 3.6))
    plt.semilogy(df['Ecm'], 100*df['p90'], label=r'$100\times \tan(p90(\theta))$ in LXe')
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Deviation [%]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

def plot_deflection(df, out='deflection_vs_E_8.png'):
    if 'deflect_p90_mm' not in df.columns: return
    plt.figure(figsize=(5.8, 3.6))
    plt.plot(df['Ecm'], df['deflect_p90_mm'])
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Drift deflection p90 [mm]')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()



# ================== OPTIMIZATION (single source of truth) ==================
import numpy as np
from scipy.optimize import minimize

def drift_metrics_rel(Er, Ez, mask):
    """
    Like drift_metrics, but the Ez floor is tied to the actual median Ez,
    avoiding artificially inflating Ez to shrink Er/Ez.
    Returns: (median_Ez, p90_ratio, inhomogeneity)
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
    floor  = max(0.05*Ed_med, float(np.percentile(Ez_abs,10)), 1.0)
    ratio  = Er_abs / np.clip(Ez_abs, floor, None)
    p90    = float(np.quantile(ratio, 0.90))
    inhEz  = float(np.quantile(np.abs(Ez_abs-Ed_med)/max(Ed_med,1.0), 0.68))
    return Ed_med, p90, inhEz


def solve_cathode_for_E0(
    Bstack, Vg, extraction_V, mask_lxe, E0_Vm,
    base_V, ring_names, vmin, vmax,
    maxit=30, tol=50.0, ring_deltas=None,
    drift_sign=-1,          # -1: Ez<0 en LXe
    enforce_order=True,     # fuerza Vc < Vg
    eps_order=10.0,         # margen mínimo (V)
    enforce_sign=True,      # evita soluciones con signo de Ez incorrecto
    eps_sign=25.0           # |Ez| mínimo (V/m) para considerar “con signo”
):
    """
    Resuelve Vc tal que mediana(Ez) ~= drift_sign * E0_Vm (en ROI_LXe),
    forzando orden físico Vc < Vg y el signo del drift.
    Devuelve (Vc_opt, hit_bound, Ed_median_signed).
    """
    target = drift_sign * E0_Vm

    def Ed_of(Vc):
        V = dict(base_V)
        V['gate']  = Vg
        V['cath']  = Vc
        V['anode'] = Vg + extraction_V
        if (ring_deltas is not None) and ring_names:
            # ring_deltas son pesos -> convertir a voltajes mediante segmentos
            names_bu, w_pos = segment_weights_from_positions(ring_names, RING_Z, z_cath=Z_CATH, z_gate=Z_GATE)
            # si ring_deltas es None usar w_pos; si viene, úsalo como pesos modificados
            w_use = np.asarray(ring_deltas, float)
            if w_use.shape[0] != w_pos.shape[0]:
                # protección: si no cuadra, usa posiciones reales
                w_use = w_pos
            V.update(rings_from_segment_weights(Vc, Vg, names_bu, w_use))
        else:
            # baseline por posición real (recomendado)
            V.update(rings_from_positions(Vc, Vg, ring_names, RING_Z, z_cath=Z_CATH, z_gate=Z_GATE))

        _, _, Er, Ez = superpose_fast(Bstack, V)
        return median_signed_Ez(Ez, mask_lxe)


    # Si pedimos orden físico, recorta el dominio de búsqueda
    if enforce_order:
        vmax = min(vmax, Vg - eps_order)

    # Valores en extremos
    Ea, Eb = Ed_of(vmin), Ed_of(vmax)

    # Si el target no cae entre Ea y Eb, quédate con el extremo más cercano
    if not (min(Ea, Eb) <= target <= max(Ea, Eb)):
        Vc_best = vmin if abs(Ea - target) < abs(Eb - target) else vmax
        Ed_best = Ed_of(Vc_best)
        hit = True
        # si además el signo está mal y lo fuerzas, empuja hacia el lado correcto
        if enforce_sign and np.sign(Ed_best) != np.sign(target):
            Vc_best = vmax  # con orden físico ya es el lado “menos campo”
            Ed_best = Ed_of(Vc_best)
        return Vc_best, hit, Ed_best

    # Bisección
    a, b, Ea, Eb = vmin, vmax, Ea, Eb
    for _ in range(maxit):
        m = 0.5*(a+b)
        Em = Ed_of(m)
        if abs(Em - target) < tol:
            # chequeos finales de signo/orden
            hit = False
            if enforce_sign and (np.sign(Em) != np.sign(target) or abs(Em) < eps_sign):
                hit = True
            return m, hit, Em
        # avanza
        if (Ea <= target <= Em) or (Em <= target <= Ea):
            b, Eb = m, Em
        else:
            a, Ea = m, Em

    # salida tras maxit
    m = 0.5*(a+b); Em = Ed_of(m)
    hit = abs(Em - target) >= tol
    if enforce_sign and (np.sign(Em) != np.sign(target) or abs(Em) < eps_sign):
        hit = True
    return m, hit, Em

def ring_weights_from_power_alpha(ring_names, alpha):
    """
    Devuelve el vector de pesos (n_rings+1) tal que las posiciones de los anillos
    siguen t_i**alpha, con t_i = (nR+1-i)/(nR+1), i=1..nR (ring_1 es el más cercano al gate).
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
    vg_seeds=(-2000.0, -1000.0, -300.0, -150.0, 0.0, +150.0, +300.0, +1000.0, 2000.0),  # multi-inicio en V_gate
    leak_band_mm=(2.0, 0.2),      # banda ancha bajo gate: (alto, bajo) en mm
    w_leak=1.20, w_hot=0.20, w_rev=0.10,  # pesos fuga/hot/reverse
    soft_anchor_k=0.03,           # anclaje suave de Vc a PMT_bot
    order_margin_V=0.0, metric='angle_bal', rings_top_factor= None, rings_power_alpha=None         
):
    """
    L-BFGS-B estable:
      - V_cath se resuelve por bisección (con orden físico y signo).
      - Se usan varias semillas de V_gate y nos quedamos con la mejor.
      - Penalizaciones “absolutas”:
          * fuga (Er/Ez) bajo el gate, banda ancha
          * hotspot y campo reverso bajo el cátodo (referencia PMT_bot)
          * anclaje suave (Vc ~ V_PMT_bot) para romper gauge inferior
      - ¡Sin premios por Vg=0!
    """
    r, z = Bstack['r'], Bstack['z']
    M_LXE = roi_mask(r, z, *roi_lxe)

    # Bandas “absolutas”
    dz_hi, dz_lo = leak_band_mm[0]*1e-3, leak_band_mm[1]*1e-3
    ROI_TOP_WIDE = (ROI_R[0], ROI_R[1], d_gate - dz_hi, d_gate - dz_lo)
    M_TOP = roi_mask(r, z, *ROI_TOP_WIDE)
    M_BELOW = roi_mask(r, z, *ROI_BELOW_CATH)

    if not np.any(M_TOP):   print("[WARN] Banda de fuga no intercepta la malla; amplía 'leak_band_mm'.")
    if not np.any(M_BELOW): print("[WARN] Banda bajo cátodo no intercepta la malla; revisa ROI_BELOW_CATH.")

    ring_names = list_rings_from_basis(Bstack); nR = len(ring_names)
    vmin_c, vmax_c = BOUNDS['cath']; vmin_g, vmax_g = BOUNDS['gate']

    rows = []
    prev_theta = None

    for E0_cm in E_targets_Vcm:
        E0 = float(E0_cm)*100.0  # V/m

        # ----- bounds y plantilla de theta -----
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
            th = [float(seed_vg)]
            if opt_pmts: th += [PMT_TOP_V, PMT_BOT_V]
            if opt_rings: th += [1.0]*(nR+1)
            if opt_offset: th += [0.0]
            return np.array(th, float)

        # semillas de Vg (respeta límite de ánodo)
        seeds = []
        if prev_theta is not None:
            seeds.append(float(prev_theta[0]))

        for vg0 in vg_seeds:
            vg0 = np.clip(vg0, vmin_g, min(vmax_g, GATE_MAX_BY_ANODE))
            if (vg0 + EXTRACTION) <= ANODE_MAX:
                seeds.append(vg0)
 
        best_res, best_theta = None, None
        for vg0 in seeds:
            th0 = make_theta(vg0)
            # pequeña inercia con la solución previa
            if prev_theta is not None:
                k = min(len(th0), len(prev_theta))
                th0[:k] = 0.7*th0[:k] + 0.3*prev_theta[:k]

            def cost_theta(theta):
                j = 0
                Vg = float(theta[j]); j += 1
                V = {}
                if opt_pmts:
                    V['pmt_top'] = float(theta[j]); V['pmt_bot'] = float(theta[j+1]); j += 2
                else:
                    V['pmt_top'] = PMT_TOP_V; V['pmt_bot'] = PMT_BOT_V

                if opt_rings:
                    deltas = np.clip(theta[j:j+nR+1], 1e-4, None)
                    j += (nR + 1)
                elif (rings_top_factor is not None) and ring_names:
                    deltas = ring_weights_uniform_plus_top(ring_names, top_factor=rings_top_factor)
                elif (rings_power_alpha is not None) and ring_names:
                    deltas = ring_weights_from_power_alpha(ring_names, rings_power_alpha)
                else:
                    # rampa lineal
                    deltas = np.ones(nR + 1, float)

                # Vc que fija el drift (con orden físico)
                Vc, hit_bound, _ = solve_cathode_for_E0(
                    Bstack, Vg, EXTRACTION, M_LXE, E0,
                    V, ring_names, vmin_c, vmax_c, tol=50.0,
                    ring_deltas=deltas, drift_sign=-1,
                    enforce_order=True, eps_order=order_margin_V,
                    enforce_sign=True, eps_sign=25.0
                )

                V.update(rings_from_positions(Vc, Vg, ring_names, deltas))

                V['gate']  = Vg
                V['cath']  = Vc
                V['anode'] = Vg + EXTRACTION

                # offset común (opcional)
                delta = float(theta[j]) if opt_offset else 0.0
                if opt_offset:
                    V['gate']  += delta; V['cath'] += delta; V['anode'] += delta
                    for name in ring_names: V[name] += delta

                # límites HV
                pen_lim = 0.0
                pen_lim += 0.0 if BOUNDS['gate'][0] <= V['gate'] <= BOUNDS['gate'][1] else 1.0
                pen_lim += 0.0 if BOUNDS['cath'][0] <= V['cath'] <= BOUNDS['cath'][1] else 1.0
                pen_lim += 0.0 if (V['anode'] <= ANODE_MAX) else 1.0

                # campo y métricas
                _, _, Er, Ez = superpose_fast(Bstack, V)
                Ed_med, p90, inhEz = drift_metrics_rel(Er, Ez, M_LXE)
                theta90 = theta_p90_deg(Er, Ez, M_LXE)
                ratio90 = np.tan(np.deg2rad(theta90)) 

                if metric == 'angle':
                    theta90 = theta_p90_deg(Er, Ez, M_LXE)
                    main = np.tan(np.deg2rad(theta90))  # ~Er/|Ez|
                elif metric == 'deflection':
                    H = (d_gate - d_cathode - 2*MARGIN_Z)  # altura efectiva de deriva (m)
                    dr90_mm = deflection_p90_mm(Er, Ez, r, z, M_LXE)
                    main = (dr90_mm/1000.0) / max(H, 1e-6)  # adimensional ~tan(ángulo medio)

                elif metric == 'rms':
                    Er_sel = Er[M_LXE]; Ez_sel = Ez[M_LXE]
                    finite = np.isfinite(Er_sel) & np.isfinite(Ez_sel)
                    Er_abs = np.abs(Er_sel[finite]); Ez_abs = np.abs(Ez_sel[finite])
                    floor  = max(1.0, np.percentile(Ez_abs, 10))
                    ratio  = Er_abs / np.clip(Ez_abs, floor, None)
                    main   = float(np.sqrt(np.nanmean(ratio**2)))

                else:  # 'angle_bal' (por defecto)
                    theta90 = theta_p90_deg_balanced(Er, Ez, M_LXE, z)
                    main = np.tan(np.deg2rad(theta90))

                # clava E0 y penaliza límites
                pen_E0 = 1e3 * ((Ed_med - E0)/max(E0, 100.0))**2
                pen = pen_E0 + 1e3*pen_lim

                # penalizaciones absolutas
                leak90 = p90_ratio_band(Er, Ez, M_TOP)               # Er/Ez en banda alta LXe
                hot90  = p90_absE(Er, Ez, M_BELOW)/max(E0, 100.0)    # |E|/E0 bajo cátodo
                revfr  = frac_reversed(Ez, M_BELOW, expected_sign=-1)

                soft_anchor = soft_anchor_k * ((V['cath'] - V['pmt_bot'])/400.0)**2

                return (main + lambda_inh*inhEz + pen) 


            res = minimize(cost_theta, th0, method='L-BFGS-B',
                           bounds=bnds, options={'maxiter': 300})
            if (best_res is None) or (res.fun < best_res.fun):
                best_res, best_theta = res, res.x

        prev_theta = best_theta.copy()

        # ----- reconstrucción con la mejor semilla -----
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

        Vopt.update(rings_from_positions(Vc_opt, Vg_opt, ring_names, deltas_final))
        Vopt['gate']  = Vg_opt + (delta_opt if opt_offset else 0.0)
        Vopt['cath']  = Vc_opt + (delta_opt if opt_offset else 0.0)
        Vopt['anode'] = Vg_opt + EXTRACTION + (delta_opt if opt_offset else 0.0)

        # métricas finales
        _, _, Er, Ez = superpose_fast(Bstack, Vopt)
        Ez_abs_roi = np.abs(Ez[M_LXE]); Ed_med = float(np.nanmedian(Ez_abs_roi)) if Ez_abs_roi.size else 0.0
        theta90 = theta_p90_deg(Er, Ez, M_LXE)
        theta90_bal = theta_p90_deg_balanced(Er, Ez, M_LXE,z)
        ratio90 = np.tan(np.deg2rad(theta90))
        ratio90_bal = np.tan(np.deg2rad(theta90_bal))
        dr90_mm = deflection_p90_mm(Er, Ez, r, z, M_LXE)
        _, _, inhEz = drift_metrics_rel(Er, Ez, M_LXE)

        # guarda fila
        ring_cols = {f"V_{name}": float(Vopt.get(name, np.nan)) for name in ring_names}
        rows.append(dict(
            Ecm=E0_cm,
            V_gate=float(Vopt['gate']),
            V_cath=float(Vopt['cath']),
            V_anode=float(Vopt['anode']),
            V_pmt_top=float(Vopt['pmt_top']),
            V_pmt_bot=float(Vopt['pmt_bot']),
            p90=float(ratio90_bal if metric=='angle_bal' else
                      ratio90 if metric=='angle' else
                      (dr90_mm/1000.0)/max((d_gate-d_cathode-2*MARGIN_Z),1e-6)),
            inhEz=float(inhEz),
            Ed_kVcm=float(Ed_med/1e5),
            hit_bound=int(hit_bound),
            theta_p90_deg=float(theta90),
            theta_p90_bal_deg=float(theta90_bal),
            deflect_p90_mm=float(dr90_mm),
            **ring_cols
        ))

    return pd.DataFrame(rows)


# ============== Optional: discrete PMT grid search (-900..-1100, step 50) ==============
PMT_GRID = np.arange(-900.0, -1100.0 - 1e-6, -50.0)  # -900, ..., -1100

def optimize_with_pmt_grid(Bstack, E_targets_Vcm, roi_lxe,
                           lambda_inh=0.6, lambda_smooth=0.02,
                           use_rings=True):
    """
    For each target field, try all (top,bottom) PMT combinations on a discrete grid,
    optimize gate (and rings if requested), solve V_cath by bisection, and pick the
    combo with minimal score = p90 + 0.5*inhEz. Returns a DataFrame of best rows.
    """
    best_rows = []
    for E0_cm in E_targets_Vcm:
        best = None
        for p_top in PMT_GRID:
            for p_bot in PMT_GRID:
                old_top, old_bot = PMT_TOP_V, PMT_BOT_V
                try:
                    globals()['PMT_TOP_V'], globals()['PMT_BOT_V'] = p_top, p_bot
                    df1 = optimize_voltages_constrained(
                        Bstack, [E0_cm], roi_lxe=roi_lxe,
                        opt_pmts=False,
                        opt_rings=use_rings,
                        lambda_inh=lambda_inh,
                        lambda_smooth=lambda_smooth
                    )
                    row = dict(df1.iloc[0])
                    row['V_pmt_top'] = p_top
                    row['V_pmt_bot'] = p_bot
                    score = row['p90'] + 0.5*row['inhEz']
                    if (best is None) or (score < best[0]):
                        best = (score, row)
                finally:
                    globals()['PMT_TOP_V'], globals()['PMT_BOT_V'] = old_top, old_bot
        best_rows.append(best[1])
    return pd.DataFrame(best_rows)

def add_Egas_and_save(df, tag):
    Eg_list = []
    for _, row in df.iterrows():
        V = dict(gate=row['V_gate'], cath=row['V_cath'], anode=row['V_anode'],
                 pmt_top=row['V_pmt_top'], pmt_bot=row['V_pmt_bot'])
        Eg_list.append(median_gas_field_from_V(Bstack, V, roi_gas=ROI_GXe))  # kV/cm
    df['Egas_kVcm'] = Eg_list
    df.to_csv(f"optimal_voltages_3D_{tag}.csv", index=False)
    plot_optimal(df, out=f"optimal_voltages_vs_E_3D_{tag}_8.png")
    plot_deviation(df, out=f"deviation_vs_E_3D_{tag}_th_8.png")
    plot_deflection(df, out=f"deflection_vs_E_3D_{tag}_8.png")
    plot_deltaEz(df, out=f"deltaEz_vs_E_3D_{tag}_8.png")
    plot_theta(df,   out=f"theta_vs_E_3D_{tag}_8.png")


def plot_p90_multi(dfs, labels, out="deviation_vs_E_3D_compare_gate_2_8.png"):
    plt.figure(figsize=(6.2, 3.8))
    for df, lab in zip(dfs, labels):
        plt.semilogy(df['Ecm'], 100.0*df['p90'], label=lab)
    # si hay columna hit_bound, marca los puntos que saturan el límite de Vcath
    for df, lab in zip(dfs, labels):
        if 'hit_bound' in df.columns:
            m = df['hit_bound'] > 0
            if m.any():
                plt.scatter(df.loc[m, 'Ecm'], 100.0*df.loc[m, 'p90'],
                            s=16, marker='x')
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Deviation [%]')
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


# ===================== Sweep V_gate at fixed E_drift =====================
def sweep_gate_for_E0(Bstack, E0_cm=60.0, roi_lxe=ROI_LXe,
                      gate_min=None, gate_max=None, n_steps=121,
                      leak_band_mm=(2.0, 0.2), out_csv="gate_sweep_E60.csv",
                      out_png="gate_sweep_E60_8.png", top_factor=None, power_alpha=None):
    """
    Barre V_gate a E_drift fijo (E0_cm en V/cm). Anillos en rampa lineal.
    Guarda CSV y figura con p90(Er/Ez), inhomog(Ez) y la zona donde el solver toca límites.
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt

    r, z = Bstack['r'], Bstack['z']
    M_LXE = roi_mask(r, z, *roi_lxe)

    # Banda "fuga" más ancha bajo el gate (en mm)
    dz_hi, dz_lo = leak_band_mm[0]*1e-3, leak_band_mm[1]*1e-3
    ROI_TOP_WIDE = (ROI_R[0], ROI_R[1], d_gate - dz_hi, d_gate - dz_lo)
    M_TOP = roi_mask(r, z, *ROI_TOP_WIDE)

    ring_names = list_rings_from_basis(Bstack); nR = len(ring_names)

    # Rango de gate
    if gate_min is None: gate_min = BOUNDS['gate'][0]
    if gate_max is None: gate_max = min(BOUNDS['gate'][1], GATE_MAX_BY_ANODE)
    gate_grid = np.linspace(gate_min, gate_max, n_steps)

    rows = []
    E0 = float(E0_cm) * 100.0  # V/m

    for Vg in gate_grid:
        Va = Vg + EXTRACTION
        if Va > ANODE_MAX:
            continue  # respeta el límite del anodo

        deltas = None
        if (top_factor is not None) and (nR > 0):
            deltas = ring_weights_uniform_plus_top(ring_names, top_factor=top_factor)
        elif (power_alpha is not None) and (nR > 0):
            deltas = ring_weights_from_power_alpha(ring_names, power_alpha)
        else:
            deltas = np.ones(nR + 1, float)

        baseV = dict(pmt_top=PMT_TOP_V, pmt_bot=PMT_BOT_V)

        Vc, hit, Ed_signed = solve_cathode_for_E0(
            Bstack, Vg, EXTRACTION, M_LXE, E0,
            baseV, ring_names, BOUNDS['cath'][0], BOUNDS['cath'][1],
            tol=50.0, ring_deltas=deltas, drift_sign=-1,
            enforce_order=True, eps_order=10.0,
            enforce_sign=True,  eps_sign=25.0
        )

        # Construye tensiones (anillos en rampa lineal)
        V = {'gate': Vg, 'cath': Vc, 'anode': Va,
             'pmt_top': PMT_TOP_V, 'pmt_bot': PMT_BOT_V}

        V.update(rings_from_positions(Vc, Vg, ring_names, deltas))


        # Campo y métricas
        _, _, Er, Ez = superpose_fast(Bstack, V)
        Ed_med, p90, inhEz = drift_metrics_rel(Er, Ez, M_LXE)
        leak90 = p90_ratio_band(Er, Ez, M_TOP)
        theta90 = theta_p90_deg(Er, Ez, M_LXE)
        p90 = np.tan(np.deg2rad(theta90))


        rows.append(dict(
            V_gate=Vg, V_cath=Vc, V_anode=Va,
            p90=p90, inhEz=inhEz, score=p90 + 0.6*inhEz + 1.2*leak90,
            leak90=leak90, Ed_kVcm=Ed_med/1e5, hit_bound=int(hit),
            theta_p90_deg=float(theta90)
        ))

    df = pd.DataFrame(rows).sort_values("V_gate").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] guardado {out_csv} con {len(df)} puntos.")

    # ----------- Figura -----------
    fig, ax = plt.subplots(figsize=(6.2,3.8))
    ax.semilogy(df['V_gate'], 100.0*df['p90'], label='p90: $E_r/E_z$ (LXe)')
    ax.semilogy(df['V_gate'], 100.0*df['inhEz'], label='inhomog $E_z$ (P90)')
    ax.set_xlabel(r'$V_{\rm gate}$ [V]')
    ax.set_ylabel('Deviation [%]')
    ax.grid(True, which='both', ls=':')
    ax.legend(loc='best')

    # marca puntos donde el solver pega en límites o cambia el orden
    m = df['hit_bound'] > 0
    if m.any():
        ax.scatter(df.loc[m,'V_gate'], 100.0*df.loc[m,'p90'], s=18, c='k', marker='x', label='hit_bound')

    # eje secundario: V_cath para ver el orden Vc<Vg
    ax2 = ax.twinx()
    ax2.plot(df['V_gate'], df['V_cath']/1e3, ls='--')
    ax2.set_ylabel(r'$V_{\rm cath}$ [kV]')
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] guardado {out_png}")

    return df


def theta_p90_deg(Er, Ez, mask):
    """
    p90 del ángulo de deriva respecto al eje z: theta = arctan(|Er|/|Ez|) en grados.
    Usa un 'floor' para Ez (como en las otras métricas) para evitar mala condición a campo muy bajo.
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

def theta_p90_deg_balanced(Er, Ez, mask, z):
    """
    Agrega por trayectoria: para cada r (columna) toma la mediana en z de theta,
    y luego el p90 entre columnas. Mucho más estable a bajo campo.
    """
    idx = np.where(mask)
    if idx[0].size == 0:
        return 0.0
    Er_sel = np.where(mask, Er, np.nan)
    Ez_sel = np.where(mask, Ez, np.nan)

    # floor de Ez para evitar mala condición
    Ez_abs = np.abs(Ez_sel)
    floor  = np.nanmax([1.0, np.nanpercentile(Ez_abs, 10)])
    ratio  = np.abs(Er_sel) / np.clip(Ez_abs, floor, None)
    theta  = np.arctan(ratio)  # rad

    # mediana en z por columna (ignora NaNs)
    theta_med_z = np.nanmedian(theta, axis=0)  # (Nr,)

    # p90 entre columnas radiales
    th90 = float(np.nanquantile(np.abs(theta_med_z), 0.90))
    return np.degrees(th90)

def deflection_p90_mm(Er, Ez, r, z, mask):
    """
    Integra |Er/Ez| a lo largo de z (trapecios) para cada r dentro de la ROI.
    Devuelve p90 de |Δr| en milímetros.
    """
    Nz, Nr = Ez.shape
    Ez_abs = np.abs(Ez)
    floor  = max(1.0, np.nanpercentile(Ez_abs[mask], 10)) if np.any(mask) else 1.0
    ratio  = np.abs(Er) / np.clip(Ez_abs, floor, None)  # ~tan(theta)

    # Δz por fila (z puede no ser uniforme)
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
        # integración por trapecios en puntos contiguos
        rj_mid = 0.5*(rj[:-1] + rj[1:])
        dz_mid = 0.5*(dzj[:-1] + dzj[1:])
        dr = np.nansum(rj_mid * dz_mid)  # en metros
        dr_list.append(abs(dr)*1000.0)   # mm

    if not dr_list:
        return 0.0
    return float(np.nanquantile(dr_list, 0.90))


def plot_deltaEz(df, out='deltaEz_vs_E_8.png'):
    """
    Grafica p90(delta E_z) = inhEz (en %) vs campo objetivo.
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


def plot_theta(df, out='theta_vs_E_8.png'):
    """
    Grafica p90 del ángulo de deriva (en grados) vs campo objetivo.
    Usa la versión 'balanced' (más robusta a bajo campo).
    """
    if 'theta_p90_bal_deg' not in df.columns:
        print("[WARN] 'theta_p90_bal_deg' not found in df; skipping plot.")
        return
    plt.figure(figsize=(5.8, 3.6))
    plt.plot(df['Ecm'], df['theta_p90_bal_deg'], label=r'$p90(\theta)$ (deg)')
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Drift angle p90 [deg]')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

def plot_deltaEz_multi(dfs, labels, out="deltaEz_vs_E_multi_8.png"):
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


def plot_theta_multi(dfs, labels, out="theta_vs_E_multi_8.png"):
    plt.figure(figsize=(6.2, 3.8))
    for df, lab in zip(dfs, labels):
        if 'theta_p90_deg' in df.columns:
            plt.plot(df['Ecm'], df['theta_p90_deg'], label=lab)
    plt.xlabel('Median drift field [V/cm]')
    plt.ylabel('Drift angle p90 [deg]')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def experiment_top_ring(Bstack, E0_cm=60.0,
                        factors=(1.00, 1.05, 1.10, 1.12)):
    """
    Para cada 'top_factor' barre V_gate, resuelve V_cath, y guarda CSV/PNG.
    Devuelve una tabla con el mejor punto (mínimo score) de cada factor.
    """
    import pandas as pd
    ring_names = list_rings_from_basis(Bstack)
    tagE = f"E{int(round(E0_cm))}"

    best_rows = []
    for f in factors:
        csv = f"gate_sweep_{tagE}_topF_{str(f).replace('.','p')}.csv"
        png = f"gate_sweep_{tagE}_topF_{str(f).replace('.','p')}_8.png"
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
    print(f"[OK] guardado best_top_ring_{tagE}.csv")
    return out

def experiment_power_alpha(Bstack, E0_cm=60.0,
                           alphas=(0.7, 0.9, 1.0, 1.2, 1.5, 2.0)):
    """
    Para cada 'alpha' barre V_gate, resuelve V_cath y guarda CSV/PNG.
    Igual que experiment_top_ring, pero usando rampa t**alpha.
    """
    import pandas as pd
    tagE = f"E{int(round(E0_cm))}"
    best_rows = []
    for a in alphas:
        csv = f"gate_sweep_{tagE}_alpha_{str(a).replace('.','p')}.csv"
        png = f"gate_sweep_{tagE}_alpha_{str(a).replace('.','p')}_8.png"
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
    print(f"[OK] guardado best_power_alpha_{tagE}.csv")
    return out

def optimize_multi_alpha(Bstack, E_targets_Vcm, roi_lxe, alphas):
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
    plot_p90_multi(dfs, labels, out="deviation_vs_E_3D_MULTI_poweralpha_8.png")
    plot_deltaEz_multi(dfs, labels, out="deltaEz_vs_E_3D_MULTI_poweralpha_8.png")
    plot_theta_multi(dfs, labels, out="theta_vs_E_3D_MULTI_poweralpha_8.png")
    return dfs




# ===================== Main Script =====================
if __name__ == "__main__":
    # 1) Load 1V electrode basis from superposition_3D.csv using cache
    Bstack = load_basis_cache(PATH_BASIS, Nr=NR_BINS, rmax=RMAX_FOR_BIN)
    r, z = Bstack['r'], Bstack['z']
    print(">>> Loaded electrode basis:", Bstack['names'])

    #df_sweep_60 = sweep_gate_for_E0(Bstack, E0_cm=60.0,
    #                            gate_min=max(-800.0, BOUNDS['gate'][0]),
    #                            gate_max=min(+1200.0, GATE_MAX_BY_ANODE),
    #                            n_steps=121,
    #                            out_csv="gate_sweep_E60.csv",
    #                            out_png="gate_sweep_E60_8.png")


    # 2) Optional validation: compare against direct simulation from example_3D.csv
    if DO_VALIDATION and os.path.exists(PATH_EXAMPLES):
        try:
            direct = load_direct_on_basis_grid_interp(PATH_EXAMPLES, Bstack, rmax=RMAX_FOR_BIN)
            # Reference voltages: gate=0, cathode=-4000, anode=+4000, fixed PMTs, ramped rings
            Vref = Vdict_for_E(gate_voltage=0.0,
                   pmt_top=PMT_TOP_V, pmt_bottom=PMT_BOT_V,
                   include_rings=True, basis_like=Bstack, ring_positions=RING_Z)

            plot_diff_map(Bstack, Vref, direct, full_extent=True, out='/Users/elenamunozrivas/Desktop/superpos_3D_2_8.png')
            print(">>> Validation plot saved: superpos_8.png")
        except Exception as e:
            print(f"[WARNING] Validation skipped: {e}")

    # 3) Voltage optimization in your drift field range (0–1300 V/cm)
    targets = np.linspace(0.0, 1300.0, 131)  # V/cm


    # --- variantes con ring_1 independiente ---

    best_alpha = experiment_power_alpha(Bstack, E0_cm=60.0,
                                        alphas=(0.7, 0.9, 1.0, 1.2, 1.5, 2.0))

    # (B) Curvas vs E comparando α:
    dfs_pow = optimize_multi_alpha(Bstack, targets, ROI_LXe,
                                   alphas=(0.7, 1.0, 1.2, 1.5, 2.0))


    factors = [0.7, 1.00, 1.2, 1.5, 2.0]  # 1.00=control (uniforme), >1.0 acerca ring_1 al gate
    
    dfs_last = []
    labels_last = []
    
    for f in factors:
        tag = f"lastring_f{str(f).replace('.','p')}_7rings_roi"
        df_last = optimize_voltages_constrained(
            Bstack, targets, roi_lxe=ROI_LXe,
            opt_pmts=False, opt_rings=False,
            lambda_inh=0.6, lambda_smooth=0.02,
            opt_offset=False, delta_bounds=(-1000.0, 1000.0),
            rings_top_factor=f            # <--- clave
        )
        add_Egas_and_save(df_last, tag)
        dfs_last.append(df_last)
        labels_last.append(f"last ring ×{f}")

    # Plots múltiples “como los tuyos”, pero comparando los factores:
    plot_p90_multi(dfs_last,    labels_last, out="deviation_vs_E_3D_MULTI_lastring_factors_8.png")
    plot_deltaEz_multi(dfs_last, labels_last, out="deltaEz_vs_E_3D_MULTI_lastring_factors_8.png")
    plot_theta_multi(dfs_last,   labels_last, out="theta_vs_E_3D_MULTI_lastring_factors_8.png")




    # Optimization without PMTs, adjusting rings
    df_opt_false_all = optimize_voltages_constrained(
        Bstack, targets, roi_lxe=ROI_LXe,
        opt_pmts=False,       
        opt_rings=False,      
        lambda_inh=0.6,
        lambda_smooth=0.02,
        opt_offset=False,                   
        delta_bounds=(-1000.0, 1000.0)
    )

    add_Egas_and_save(df_opt_false_all, "opt_false_all_7rings_roi")


    row = df_opt_false_all.iloc[(df['Ecm'] - 60).abs().argmin()]

    Vopt = dict(
        gate=row['V_gate'],
        cath=row['V_cath'],
        anode=row['V_anode'],
        pmt_top=row['V_pmt_top'],
        pmt_bot=row['V_pmt_bot']
    )

    # Llama al plot
    plot_deltaE_map(Bstack, Vopt, roi=ROI_LXe, out="deltaE_map_E60.png")




    df_opt_pmt = optimize_voltages_constrained(
        Bstack, targets, roi_lxe=ROI_LXe,
        opt_pmts=True, opt_rings=False
    )
    add_Egas_and_save(df_opt_pmt, "opt_pmt_7rings_roi")

    df_opt_rings = optimize_voltages_constrained(
        Bstack, targets, roi_lxe=ROI_LXe,
        opt_pmts=False, opt_rings= True
    )
    add_Egas_and_save(df_opt_rings, "opt_rings_7rings_roi")
    
    df_opt_pmt_rings = optimize_voltages_constrained(
        Bstack, targets, roi_lxe=ROI_LXe,
        opt_pmts=True, opt_rings= True
    )
    add_Egas_and_save(df_opt_pmt_rings, "opt_pmt_rings_7rings_roi")

    # Escoge tres de tus DataFrames
    dfs    = [df_opt_false_all , df_opt_pmt, df_opt_rings, df_opt_pmt_rings ]  # ejemplo
    labels = ["base (rings OFF, PMTs OFF)", "PMTs ON", "rings ON", "PMTs+rings ON"]

    # p90(Er/Ez) ya lo cubre tu plot_p90_multi:
    plot_p90_multi(dfs, labels, out="deviation_vs_E_3D_MULTI_lastring_8.png")

    # Añade los nuevos “multi”
    plot_deltaEz_multi(dfs, labels, out="deltaEz_vs_E_3D_MULTI_lastring_8.png")
    plot_theta_multi(dfs,   labels, out="theta_vs_E_3D_MULTI_lastring_8.png")




    old = BOUNDS['gate']
    BOUNDS['gate'] = ( -1000.0,  -1000.0)  # -1000 V fijo
    df_gate_m1000 = optimize_voltages_constrained(Bstack, targets, roi_lxe=ROI_LXe,
                                            opt_pmts=False, opt_rings= False,
                                            lambda_inh=0.6, lambda_smooth=0.05)
    BOUNDS['gate'] = (0.0, 0.0)  # 0 V fijo
    df_gate_0 = optimize_voltages_constrained(Bstack, targets, roi_lxe=ROI_LXe,
                                            opt_pmts=False, opt_rings= False,
                                            lambda_inh=0.6, lambda_smooth=0.05)
    BOUNDS['gate'] = ( 1000.0,  1000.0)  # +1000 V fijo
    df_gate_p1000 = optimize_voltages_constrained(Bstack, targets, roi_lxe=ROI_LXe,
                                            opt_pmts=False, opt_rings= False,
                                            lambda_inh=0.6, lambda_smooth=0.05)

    BOUNDS['gate'] = old


    add_Egas_and_save(df_opt,       "freegate")
    add_Egas_and_save(df_gate_p1000,  "gatep1000")
    add_Egas_and_save(df_gate_m1000,  "gatem1000")

    plot_p90_multi(
        [df_gate_m1000, df_gate_0, df_gate_p1000],
        labels=[r'$V_{\rm gate}=-1000$ V', r'$V_{\rm gate}=0$ V', r'$V_{\rm gate}=+1000$ V'],
        out="deviation_vs_E_3D_compare_gate_8.png"
    )
    print(">>> Saved: deviation_vs_E_3D_compare_gate_8.png")



