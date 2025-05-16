import os
from typing import Any
import ceci

from rail.core.stage import RailStage, RailPipeline

from rail.utils.recalib_algo_library import RECALIB_ALGORITHMS

DUMMY_ALGOS: dict[str, Any] = dict(trainz=None)


class EstimateRecalibPipeline(RailPipeline):

    default_input_dict={
        'input':'dummy.in',
    }

    def __init__(self, algorithms: dict|None=None, recalib_algos: dict|None=None, models_dir: str='.'):

        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = DUMMY_ALGOS.copy()

        if recalib_algos is None:
            recalib_algos = RECALIB_ALGORITHMS.copy()

        for algo_ in algorithms.keys():

            input_path = f"output_estimate_{algo_}.hdf5"
            self.default_input_dict[f"input_{algo_}"] = os.path.join(models_dir, input_path)
            
            for key, val in recalib_algos.items():

                the_class = ceci.PipelineStage.get_stage(val['Estimate'], val['Module'])
                the_estimator = the_class.make_and_connect(
                    name=f'estimate_{algo_}_{key}',
                    aliases=dict(
                        model=f"model_{algo_}_{key}",
                        input=f"input_{algo_}",
                    ),
                    hdf5_groupname='',
                )
                model_path = f'model_inform_{algo_}_{key}.pkl'
                self.default_input_dict[f"model_{algo_}_{key}"] = os.path.join(models_dir, model_path)
                self.add_stage(the_estimator)
