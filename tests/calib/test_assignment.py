import os

import pytest
import qp
from rail.core.stage import RailStage

from rail.calib.cell_assignment import PZMaxCellPCellAssigner, PZModeCellAssigner

pzdata_path = "tests/output_estimate_knn.hdf5"


@pytest.mark.parametrize(
    "name,assign_class",
    [
        ("pz_mode", PZModeCellAssigner),
        ("max_p", PZMaxCellPCellAssigner),
    ],
)
def test_assignment(get_data: int, name: str, assign_class: type[RailStage]) -> None:

    assert get_data == 0

    pzdata = qp.read(pzdata_path) # DS.read_file("pzdata", QPHandle, path=pzdata_path)

    assigner = assign_class.make_stage(name=f"assign_{name}")

    assignment = assigner(pzdata)

    os.remove(
        assigner.get_output(assigner.get_aliased_tag("assignment"), final_name=True)
    )
