import os
import pytest

from rail.core.stage import RailStage
from rail.core.data import QPHandle, TableHandle
from rail.calib.cell_assignment_estimator import (
    PZModeCellAssignmentPzInformer,
    PZMaxCellPCellAssignmentPzInformer,
    PZModeCellAssignmentPzEstimator,
    PZMaxCellPCellAssignmentPzEstimator,
)


testdata_path = 'tests/dc2_run2p2i_dr6_test_dered.hdf5'
pzdata_path = 'tests/output_estimate_knn.hdf5'


@pytest.mark.parametrize(
    "name,inform_class,estimate_class",
    [
        ("pz_mode", PZModeCellAssignmentPzInformer, PZModeCellAssignmentPzEstimator),
        ("max_p", PZMaxCellPCellAssignmentPzInformer, PZMaxCellPCellAssignmentPzEstimator),
    ],
)
def test_recalib(get_data: int, name: str, inform_class: type[RailStage], estimate_class: type[RailStage]) -> None:

    assert get_data == 0
    
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

    testdata = DS.read_file("testdata", TableHandle, path=testdata_path)
    pzdata = DS.read_file("pzdata", QPHandle, path=pzdata_path)
    
    informer = inform_class.make_stage(
        name=f'inform_{name}',
        redshift_col='redshift_true',
    )
    
    model = informer.inform(pzdata, testdata)['model']

    estimator = estimate_class.make_stage(        
        name=f'estimate_{name}',
        model=model,
    )

    estimation = estimator.estimate(pzdata)

    os.remove(informer.get_output(informer.get_aliased_tag("model"), final_name=True))
    os.remove(informer.get_output(informer.get_aliased_tag("assignment"), final_name=True))
    os.remove(estimator.get_output(informer.get_aliased_tag("output"), final_name=True))

