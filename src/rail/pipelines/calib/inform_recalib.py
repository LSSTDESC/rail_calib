import ceci

from rail.core.stage import RailStage, RailPipeline

from rail.utils.recalib_algo_library import RECALIB_ALGORITHMS


class InformRecalibPipeline(RailPipeline):

    default_input_dict={
        'input':'dummy.in',
        'truth':'dummy.in',
    }

    def __init__(self, recalib_algos: dict|None=None):

        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if recalib_algos is None:
            recalib_algos = RECALIB_ALGORITHMS.copy()
            
        for key, val in recalib_algos.items():
            the_class = ceci.PipelineStage.get_stage(val['Inform'], val['Module'])
            the_informer = the_class.make_and_connect(
                name=f'inform_{key}',
                aliases=dict(
                    input='input',
                    truth='truth',
                ),
                hdf5_groupname='',
            )
            self.add_stage(the_informer)

