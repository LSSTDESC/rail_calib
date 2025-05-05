import os
import pytest

from rail.core.stage import RailStage
from rail.core.data import QPHandle, TableHandle
from rail.calib.cell_assignment import PZModeCellAssigner, PZMaxCellPCellAssigner

pzdata_path = 'tests/output_estimate_knn.hdf5'


@pytest.mark.parametrize(
    "name,assign_class",
    [
        ("pz_mode", PZModeCellAssigner),
        ("max_p", PZMaxCellPCellAssigner),
    ],
)
def test_assignment(get_data: int, name: str, assign_class: type[RailStage]) -> None:

    assert get_data == 0
    
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

    pzdata = DS.read_file("pzdata", QPHandle, path=pzdata_path)
    
    assigner = assign_class.make_stage(name=f"assign_{name}")

    assignment = assigner(pzdata)

    os.remove(assigner.get_output(assigner.get_aliased_tag("assignment"), final_name=True))
