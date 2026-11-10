"""Independent controls for confidence repair and immutable response contracts."""
import numpy as np
from evaluation.document_box_neighbor_20261027 import neighbor


def test_unsupported_jump_and_no_anchor():
    """@luna-comment
    Input: Continuous distribution with one false lower-rail spike, then all false rails.
    Output: Spike repairs toward10; trusted nodes remain exact; no-anchor case unchanged.
    Key Steps: Supply synthetic descriptors independent of an image, inspect repair/support.
    Key Params:5x5nodes, uniformQ1=10, median100; no semantic text fixture.
    State/Dependencies: Pure NumPy and new repair function; no external files.
    """
    nodes=np.arange(5)*8
    q=np.broadcast_to(np.array([10,15,20,60,100,120,140],np.float32),(5,5,7)).copy()
    low=np.full((5,5),10.,np.float32);high=np.full_like(low,150.)
    low[2,2]=95
    fixed,info=neighbor.repair(low,high,q,q[:,:,0],nodes)
    assert fixed[2,2]<11 and info['corrected_nodes']==1
    np.testing.assert_array_equal(fixed[0],low[0])
    false=np.full_like(low,95)
    fixed,info=neighbor.repair(false,high,q,q[:,:,0],nodes)
    np.testing.assert_array_equal(fixed,false)
    assert info['unresolved_nodes']==25


def test_material_step_and_constant():
    """@luna-comment
    Input: Two valid brightness populations with a sharp material boundary and a flat field.
    Output: Correct rails remain exact across the boundary; constant mapping identity.
    Key Steps: Shift all descriptors with actual rails, repair and verify unchanged arrays.
    Key Params: No accidental cross-material dark-rail propagation permitted in this control.
    State/Dependencies: NumPy and candidate tables; no semantic classification.
    """
    nodes=np.arange(5)*8
    low=np.full((5,5),10.,np.float32);low[:,3:]=100
    q=low[:,:,None]+np.array([0,2,5,10,20,30,40],np.float32)
    fixed,_=neighbor.repair(low,low+80,q,q[:,:,0],nodes)
    np.testing.assert_array_equal(fixed,low)
    lut=neighbor.tables(np.full((1,1),90.),np.full((1,1),90.))
    np.testing.assert_array_equal(lut[0,0],np.arange(256))


def test_monotone_and_repair_direction():
    """@luna-comment
    Input: Original erroneousL90and repairedL10with fixedH180.
    Output: Monotone LUTs; weak repair lies between baseline/full for dark code1.
    Key Steps: Evaluate independent rail strengths and compare dark responses.
    Key Params: Existing upper rail and STANDARDtail parameters are fixed.
    State/Dependencies: No native statistics or runtime file operations.
    """
    values=[]
    for low in [90.,50.,10.]:
        lut=neighbor.tables(np.array([[low]]),np.array([[180.]]))
        assert np.all(np.diff(lut[0,0])>=0)
        values.append(lut[0,0,1])
    assert values[0]>values[1]>values[2]
