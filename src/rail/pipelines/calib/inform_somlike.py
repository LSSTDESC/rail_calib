from ceci import PipelineStage

from rail.core.stage import RailStage, RailPipeline
from rail.utils.catalog_utils import CatalogConfigBase
from rail.utils.recalib_algo_library import DEFAULT_PZ_ALGORITHM


class InformSomlikePipeline(RailPipeline):
    default_input_dict = {
        "input_deep_data": "dummy.in",
        "input_wide_data": "dummy.in",
    }

    def __init__(
        self,
        wide_informer: dict = DEFAULT_PZ_ALGORITHM,
        deep_informer: dict = DEFAULT_PZ_ALGORITHM,        
        wide_catalog_tag: str = "SompzWideTestCatalogConfig",
        deep_catalog_tag: str = "SompzDeepTestCatalogConfig",
        catalog_module: str = "rail.sompz.utils",
    ):
        RailPipeline.__init__(self)

        wide_catalog_class = CatalogConfigBase.get_class(
            wide_catalog_tag, catalog_module
        )
        deep_catalog_class = CatalogConfigBase.get_class(
            deep_catalog_tag, catalog_module
        )

        deep_informer_class = PipelineStage.get_stage(deep_informer['Inform'], deep_informer['Module'])
        wide_informer_class = PipelineStage.get_stage(wide_informer['Inform'], wide_informer['Module'])

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        # 1: train the deep SOM
        CatalogConfigBase.apply(deep_catalog_class.tag)        
        self.pz_informer_deep = deep_informer_class.build(
            aliases=dict(input="input_deep_data"),
        )

        # 2: train the wide SOM
        CatalogConfigBase.apply(wide_catalog_class.tag)        
        self.pz_informer_wide = wide_informer_class.build(
            aliases=dict(input="input_wide_data"),
        )
