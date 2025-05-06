import os
from rail.utils.testing_utils import build_and_read_pipeline
import ceci

import pytest


@pytest.mark.parametrize(
    "pipeline_class",
    [
        'rail.pipelines.calib.estimate_recalib.EstimateRecalibPipeline',
        'rail.pipelines.calib.inform_recalib.InformRecalibPipeline',
        'rail.pipelines.calib.somlike_recalib.SomlikeRecalibPipeline',
        'rail.pipelines.calib.inform_somlike.InformSomlikePipeline',
    ]
)
def test_build_and_read_pipeline(pipeline_class):
    build_and_read_pipeline(pipeline_class)

