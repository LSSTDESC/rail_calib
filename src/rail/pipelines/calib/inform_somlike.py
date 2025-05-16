from typing import Any

from ceci import PipelineStage

from rail.core.stage import RailStage, RailPipeline
from rail.utils.catalog_utils import CatalogConfigBase
from rail.utils.recalib_algo_library import DEFAULT_PZ_ALGORITHM


class InformSomlikePipeline(RailPipeline):
    default_input_dict = {
        "input_spec_data": "dummy.in",
    }

    def __init__(
        self,
        algorithms: dict|None=None, 
        wide_catalog_tag: str = "SompzWideTestCatalogConfig",
        deep_catalog_tag: str = "SompzDeepTestCatalogConfig",
        catalog_module: str = "rail.sompz.utils",
    ):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = dict(trainz=DEFAULT_PZ_ALGORITHM)

        wide_catalog_class = CatalogConfigBase.get_class(
            wide_catalog_tag, catalog_module
        )
        deep_catalog_class = CatalogConfigBase.get_class(
            deep_catalog_tag, catalog_module
        )
        
        for algo_, algo_info_ in algorithms.items():

            informer_class = PipelineStage.get_stage(algo_info_['Inform'], algo_info_['Module'])

            # 1: inform for the deep sample
            CatalogConfigBase.apply(deep_catalog_class.tag)        
            pz_informer_deep = informer_class.make_stage(
                name=f"pz_informer_{algo_}_deep",
                aliases=dict(input="input_spec_data"),
            )
            self.add_stage(pz_informer_deep)
            
            # 2:  inform for the wide sample
            CatalogConfigBase.apply(wide_catalog_class.tag)        
            pz_informer_wide = informer_class.make_stage(
                name=f"pz_informer_{algo_}_wide",
                aliases=dict(input="input_spec_data"),
            )
            self.add_stage(pz_informer_wide)
