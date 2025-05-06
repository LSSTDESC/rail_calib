

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
