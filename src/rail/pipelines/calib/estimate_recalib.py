import os
import ceci

from rail.core.stage import RailStage, RailPipeline

from rail.utils.recalib_algo_library import RECALIB_ALGORITHMS


class EstimateRecalibPipeline(RailPipeline):

    default_input_dict={
        'input':'dummy.in',
    }

    def __init__(self, recalib_algos: dict|None=None, models_dir: str='.'):

        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if recalib_algos is None:
            recalib_algos = RECALIB_ALGORITHMS.copy()

        for key, val in recalib_algos.items():
            the_class = ceci.PipelineStage.get_stage(val['Estimate'], val['Module'])
            the_estimator = the_class.make_and_connect(
                name=f'inform_{key}',
                aliases=dict(model=f"model_{key}"),
                hdf5_groupname='',
            )
            model_path = f'inform_model_{key}.pkl'
            self.default_input_dict[f"model_{key}"] = os.path.join(models_dir, model_path)            
            self.add_stage(the_estimator)

