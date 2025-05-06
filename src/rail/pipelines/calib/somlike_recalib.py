from ceci import PipelineStage

from rail.utils.catalog_utils import CatalogConfigBase
from rail.core.stage import RailStage, RailPipeline

from rail.estimation.algos.sompz import SOMPZEstimatorWide, SOMPZEstimatorDeep
from rail.estimation.algos.sompz import SOMPZPzc, SOMPZPzchat, SOMPZPc_chat
from rail.estimation.algos.sompz import SOMPZTomobin, SOMPZnz

from rail.utils.recalib_algo_library import ASSIGN_ALGORITHMS, DEFAULT_PZ_ALGORITHM



bin_edges_deep = [0.0, 0.5, 1.0, 2.0, 3.0]
zbins_min_deep = 0.0
zbins_max_deep = 3.2
zbins_dz_deep = 0.02

bin_edges_tomo = [0.2, 0.6, 1.2, 1.8, 2.5]
zbins_min_tomo = 0.0
zbins_max_tomo = 3.0
zbins_dz_tomo = 0.025


class SomlikeRecalibPipeline(RailPipeline):
    default_input_dict = {
        "wide_model": "dummy.in",
        "deep_model": "dummy.in",
        "input_spec_data": "dummy.in",
        "input_deep_data": "dummy.in",
        "input_wide_data": "dummy.in",
    }

    def __init__(
        self,
        wide_estimator: dict = DEFAULT_PZ_ALGORITHM,
        deep_estimator: dict = DEFAULT_PZ_ALGORITHM,
        wide_assignment: dict = ASSIGN_ALGORITHMS['pz_mode'],
        deep_assignment: dict = ASSIGN_ALGORITHMS['pz_mode'],
        wide_catalog_tag: str = "SompzWideTestCatalogConfig",
        deep_catalog_tag: str = "SompzDeepTestCatalogConfig",
        catalog_module: str = "rail.sompz.utils",
    ):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        wide_catalog_class = CatalogConfigBase.get_class(
            wide_catalog_tag, catalog_module
        )
        deep_catalog_class = CatalogConfigBase.get_class(
            deep_catalog_tag, catalog_module
        )

        deep_estimator_class = PipelineStage.get_stage(deep_estimator['Estimate'], deep_estimator['Module'])
        deep_assignment_class = PipelineStage.get_stage(deep_assignment['Assign'], deep_assignment['Module'])
        wide_estimator_class = PipelineStage.get_stage(wide_estimator['Estimate'], wide_estimator['Module'])
        wide_assignment_class = PipelineStage.get_stage(wide_assignment['Assign'], wide_assignment['Module'])
        
        # Do the deep model assignment stuff here
        CatalogConfigBase.apply(deep_catalog_class.tag)
        
        # 1. Find the best cell mapping for all of the deep/balrog galaxies into the deep SOM
        self.pz_deepdeep_estimator = deep_estimator_class.build(
            aliases=dict(model="deep_model", input="input_deep_data"),
        )
        
        self.deepdeep_assigment = deep_assignment_class.build(
            connections=dict(
                pz_estimate=self.pz_deepdeep_estimator.io.output
            ),
        )        
        
        # 3. Find the best cell mapping for all of the spectrscopic galaxies into the deep SOM
        self.pz_deepspec_estimator = deep_estimator_class.build(
            aliases=dict(model="deep_model", input="input_spec_data"),
        )

        self.deepspec_assigment = deep_assignment_class.build(
            connections=dict(
                pz_estimate=self.pz_deepspec_estimator.io.output
            ),            
        )

        # Now do the wide model assignment stuff here
        CatalogConfigBase.apply(wide_catalog_class.tag)
        
        # 2. Find the best cell mapping for all of the deep/balrog galaxies into the wide SOM
        self.pz_deepwide_estimator = wide_estimator_class.build(
            aliases=dict(model="wide_model", input="input_deep_data"),
        )

        self.deepwide_assigment = wide_assignment_class.build(
            connections=dict(
                pz_estimate=self.pz_deepwide_estimator.io.output
            ),
        )

        # 6. Find the best cell mapping for all of the wide-field galaxies into the wide SOM
        self.pz_widewide_estimator = wide_estimator_class.build(
            aliases=dict(model="wide_model", input="input_wide_data"),
        )

        self.widewide_assigment = wide_assignment_class.build(
            connections=dict(
                pz_estimate=self.pz_widewide_estimator.io.output
            ),                        
        )

        # 8. Find the best cell mapping for all of the spectroscopic galaxies into the wide SOM
        self.pz_widespec_estimator = wide_estimator_class.build(
            aliases=dict(model="wide_model", input="input_spec_data"),
        )

        self.widespec_assigment = wide_assignment_class.build(
            connections=dict(
                pz_estimate=self.pz_widespec_estimator.io.output
            ),                                    
        )
        

        # 4. Use these cell assignments to compute the pz_c redshift histograms in deep SOM.
        # These distributions are redshift pdfs for individual deep SOM cells.
        self.som_pzc = SOMPZPzc.build(
            redshift_col="redshift",
            bin_edges=bin_edges_deep,
            zbins_min=zbins_min_deep,
            zbins_max=zbins_max_deep,
            zbins_dz=zbins_dz_deep,
            deep_groupname="",
            aliases=dict(spec_data="input_spec_data"),
            connections=dict(
                cell_deep_spec_data=self.deepspec_assigment.io.assignment
            ),
        )

        # 5. Compute the 'transfer function'.
        # The 'transfer function' weights relating deep to wide photometry.
        # These weights set the relative importance of p(z) from deep SOM cells for each
        # corresponding wide SOM cell.
        # These are traditionally made by injecting galaxies into images with Balrog.
        self.som_pcchat = SOMPZPc_chat.build(
            connections=dict(
                cell_deep_balrog_data=self.deepdeep_assigment.io.assignment,
                cell_wide_balrog_data=self.deepwide_assigment.io.assignment,
            )
        )
        
        # 7. Compute more weights.
        # These weights represent the normalized occupation fraction of each wide SOM cell
        # relative to the full sample.
        self.som_pzchat = SOMPZPzchat.build(
            bin_edges=bin_edges_tomo,
            zbins_min=zbins_min_tomo,
            zbins_max=zbins_max_tomo,
            zbins_dz=zbins_dz_tomo,
            redshift_col="redshift",
            aliases=dict(
                spec_data="input_spec_data",
            ),
            connections=dict(
                cell_deep_spec_data=self.deepspec_assigment.io.assignment,
                cell_wide_wide_data=self.widewide_assigment.io.assignment,
                pz_c=self.som_pzc.io.pz_c,
                pc_chat=self.som_pcchat.io.pc_chat,
            ),
        )
        
        # 9. Define a tomographic bin mapping
        self.som_tomobin = SOMPZTomobin.build(
            bin_edges=bin_edges_tomo,
            zbins_min=zbins_min_tomo,
            zbins_max=zbins_max_tomo,
            zbins_dz=zbins_dz_tomo,
            wide_som_size=625,
            deep_som_size=1024,
            redshift_col="redshift",
            aliases=dict(
                spec_data="input_spec_data",
            ),
            connections=dict(
                cell_deep_spec_data=self.deepspec_assigment.io.assignment,
                cell_wide_spec_data=self.widespec_assigment.io.assignment,
            ),
        )

        # 10. Assemble the final tomographic bin estimates
        self.som_nz = SOMPZnz.build(
            bin_edges=bin_edges_tomo,
            zbins_min=zbins_min_tomo,
            zbins_max=zbins_max_tomo,
            zbins_dz=zbins_dz_tomo,
            redshift_col="redshift",
            aliases=dict(
                spec_data="input_spec_data",
            ),
            connections=dict(
                cell_deep_spec_data=self.deepspec_assigment.io.assignment,
                cell_wide_wide_data=self.widewide_assigment.io.assignment,
                tomo_bins_wide=self.som_tomobin.io.tomo_bins_wide,
                pc_chat=self.som_pcchat.io.pc_chat,
            ),
        )
