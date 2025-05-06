

ASSIGN_ALGORITHMS = dict(
    pz_mode=dict(
        Assign='PZModeCellAssigner',
        Module='rail.calib.cell_assignment',
        
    ),
    pz_max_cell_p=dict(
        Assign='PZModeCellAssigner',
        Module='rail.calib.cell_assignment',
    ),
)


RECALIB_ALGORITHMS = dict(
    pz_mode=dict(
        Inform='PZModeCellAssignmentPzInformer',
        Estimate='PZModeCellAssignmentPzEstimator',
        Module='rail.calib.cell_assignment_estimator',
    ),
    pz_max_cell_p=dict(
        Inform='PZMaxCellPCellAssignmentPzInformer',
        Estimate='PZMaxCellPCellAssignmentPzEstimator',
        Module='rail.calib.cell_assignment_estimator',
    ),
)


DEFAULT_PZ_ALGORITHM = dict(
    Inform='KNearNeighInformer',
    Estimate='KNearNeighEstimator',
    Module='rail.estimation.algos.k_nearneigh',
)
