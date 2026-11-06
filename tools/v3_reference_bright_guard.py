"""Source V3 repair-evidence attenuation of the existing bright branch only."""
import numpy as np

from evaluation.document_box_neighbor_20261027 import neighbor
from tools.v3_reference_post import NODES, anchored_dark_map, node_tables, smoothstep01


def repair_evidence(original_low, low3, high, quantiles, small_tail, nodes):
    """@luna-comment Input: hash-bound original V3 fields. Output: repair amount, downward movement and evidence. Key Steps: replay frozen neighbor repair exactly, then multiply amount by smoothstep of actual rail movement. Key Params: allowance12, scale24. State/Dependencies: source-only; evidence is a new experimental hypothesis, not a semantic confidence."""
    repaired, info = neighbor.repair(original_low, high, quantiles, small_tail, nodes)
    if not np.array_equal(repaired, low3):
        raise AssertionError('Frozen neighbor repair does not reproduce cached L3')
    amount = np.asarray(info['amount'], np.float32)
    movement = np.maximum(np.asarray(original_low, np.float32)-np.asarray(low3, np.float32), 0.)
    evidence = amount*smoothstep01((movement-12.)/24.)
    if not (np.isfinite(evidence).all() and np.all((evidence >= 0.)&(evidence <= 1.))):
        raise AssertionError('Invalid repair evidence')
    return amount, movement, evidence.astype(np.float32)


def guarded_node_tables(low3, high, population, evidence, etas=(.35, 1.)):
    """@luna-comment Input: frozen source V3 node fields and actual-movement evidence. Output: old full and guarded monotone Y8 LUTs. Key Steps: keep old dark/polarity/span maps, subtract only eta times evidence-weighted bright correction. Key Params: eta.35/1, lambda1, g3/T12/support24. State/Dependencies: no model-output statistic or H transport; eta0 returns frozen old table exactly."""
    low3 = np.asarray(low3, np.float32)
    high = np.asarray(high, np.float32)
    population = np.asarray(population, np.float32)
    evidence = np.asarray(evidence, np.float32)
    expected = (len(NODES), len(NODES))
    if any(array.shape != expected for array in (low3, high, population, evidence)):
        raise ValueError('Guard field geometry mismatch')
    if not np.isfinite(evidence).all() or np.any((evidence < 0.)|(evidence > 1.)):
        raise ValueError('Invalid evidence range')
    old, old_telemetry = node_tables(low3, high, population, strengths=(1.,))
    original = old[1.]
    y = np.arange(256, dtype=np.float32)[None, None, :]
    bright = 255.-anchored_dark_map(255.-y, 255.-low3[:, :, None])
    bright_weight = smoothstep01((.5-population[:, :, None])/.25)
    span = np.maximum(high-low3, 0.)[:, :, None]
    support = span*span/(span*span+24.*24.)
    bright_correction = support*bright_weight*(bright-y)
    tables = {}
    for eta in etas:
        if not 0. <= eta <= 1.:
            raise ValueError('Eta outside [0,1]')
        if eta == 0.:
            table = original.copy()
        else:
            table = np.asarray(original-eta*evidence[:, :, None]*bright_correction, np.float32)
            exact = (evidence == 0.) | (bright_weight[:, :, 0] == 0.)
            table[exact] = original[exact]
        if not np.isfinite(table).all() or np.min(np.diff(table, axis=2)) < -1e-4:
            raise AssertionError('Guarded node table not monotone')
        if np.any(table > original+1e-4):
            raise AssertionError('Bright guard increased an old table value')
        tables[eta] = table
    telemetry = dict(old_telemetry)
    telemetry.update(evidence_active_nodes=int(np.count_nonzero(evidence)),
                     evidence_max=float(evidence.max()),
                     bright_evidence_nodes=int(np.count_nonzero((evidence > 0.)&(bright_weight[:, :, 0] > 0.))))
    return original, tables, telemetry
