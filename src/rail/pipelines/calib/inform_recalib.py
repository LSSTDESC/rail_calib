from typing import Any
import os

import ceci

from rail.core.stage import RailStage, RailPipeline
from rail.utils.recalib_algo_library import RECALIB_ALGORITHMS

DUMMY_ALGOS: dict[str, Any] = dict(trainz=None)


class InformRecalibPipeline(RailPipeline):

    default_input_dict={
        'truth':'dummy.in',
    }

    def __init__(self, algorithms: dict|None=None, recalib_algos: dict|None=None, pdfs_dir: str='.'):

        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = DUMMY_ALGOS.copy()
        
        if recalib_algos is None:
            recalib_algos = RECALIB_ALGORITHMS.copy()

        for algo_ in algorithms.keys():
            pdf_path = f'output_estimate_{algo_}.hdf5'
            self.default_input_dict[f"input_{algo_}"] = os.path.join(pdfs_dir, pdf_path)
            
            for key, val in recalib_algos.items():
                the_class = ceci.PipelineStage.get_stage(val['Inform'], val['Module'])
                the_informer = the_class.make_and_connect(
                    name=f'inform_{algo_}_{key}',
                    aliases=dict(
                        input=f'input_{algo_}',
                    ),
                    hdf5_groupname='',
                )
                self.add_stage(the_informer)

