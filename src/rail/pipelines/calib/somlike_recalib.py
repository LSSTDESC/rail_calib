import os

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
        "input_spec_data": "dummy.in",
        "input_deep_data": "dummy.in",
        "input_wide_data": "dummy.in",
    }

    def __init__(
        self,
        algorithms: dict|None=None,
        assign_algos: dict|None=None,
        models_dir: str=".",
        wide_catalog_tag: str = "SompzWideTestCatalogConfig",
        deep_catalog_tag: str = "SompzDeepTestCatalogConfig",
        catalog_module: str = "rail.sompz.utils",
    ):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if algorithms is None:
            algorithms = dict(trainz=DEFAULT_PZ_ALGORITHM)

        if assign_algos is None:
            assign_algos = ASSIGN_ALGORITHMS.copy()

        wide_catalog_class = CatalogConfigBase.get_class(
            wide_catalog_tag, catalog_module
        )
        deep_catalog_class = CatalogConfigBase.get_class(
            deep_catalog_tag, catalog_module
        )

        for algo_, algo_info_ in algorithms.items():
            estimator_class = PipelineStage.get_stage(algo_info_['Estimate'], algo_info_['Module'])

            CatalogConfigBase.apply(deep_catalog_class.tag)

            self.default_input_dict[f"model_{algo_}_deep"] = os.path.join(models_dir, f"model_pz_informer_{algo_}_deep.pkl")
            self.default_input_dict[f"model_{algo_}_wide"] = os.path.join(models_dir, f"model_pz_informer_{algo_}_wide.pkl")
            
            # 1.A estimate pz in the deep field, using the deep model
            pz_deepdeep_estimator = estimator_class.make_stage(
                name=f"pz_deepdeep_estimator_{algo_}",
                aliases=dict(model=f"model_{algo_}_deep", input="input_deep_data"),
            )
            self.add_stage(pz_deepdeep_estimator)

            # 3.A estimate pz in the spec field, using the deep model
            pz_deepspec_estimator = estimator_class.make_stage(
                name=f"pz_deepspec_estimator_{algo_}",
                aliases=dict(model=f"model_{algo_}_deep", input="input_spec_data"),
            )
            self.add_stage(pz_deepspec_estimator)


            CatalogConfigBase.apply(wide_catalog_class.tag)

            # 2.A estimate pz in the spec field, using the deep model
            pz_deepwide_estimator = estimator_class.make_stage(
                name=f"pz_deepwide_estimator_{algo_}",
                aliases=dict(model=f"model_{algo_}_wide", input="input_deep_data"),
            )
            self.add_stage(pz_deepwide_estimator)

            # 6.A estimate pz in the wide field, using the wide model
            pz_widewide_estimator = estimator_class.make_stage(
                name=f"pz_widewide_estimator_{algo_}",
                aliases=dict(model=f"model_{algo_}_wide", input="input_wide_data"),
            )
            self.add_stage(pz_widewide_estimator)

            # 8.A Find the best cell mapping for all of the spectroscopic galaxies into the wide SOM
            pz_widespec_estimator = estimator_class.make_stage(
                name=f"pz_widespec_estimator_{algo_}",
                aliases=dict(model=f"model_{algo_}_wide", input="input_spec_data"),
            )
            self.add_stage(pz_widespec_estimator)


            for assign_name_, assign_info_ in assign_algos.items():
                assignment_class = PipelineStage.get_stage(assign_info_['Assign'], assign_info_['Module'])

                # 1.B Find the best cell mapping for all of the deep/balrog galaxies into the deep SOM
                deepdeep_assigment = assignment_class.make_and_connect(
                    name=f"deepdeep_assigment_{algo_}_{assign_name_}",
                    connections=dict(
                        pz_estimate=pz_deepdeep_estimator.io.output
                    ),
                )
                self.add_stage(deepdeep_assigment)
                
                # 3.B Find the best cell mapping for all of the spectrscopic galaxies into the deep SOM
                deepspec_assigment = assignment_class.make_and_connect(
                    name=f"deepspec_assigment_{algo_}_{assign_name_}",
                    connections=dict(
                        pz_estimate=pz_deepspec_estimator.io.output
                    ),
                )
                self.add_stage(deepspec_assigment)

                # 2.B Find the best cell mapping for all of the deep/balrog galaxies into the wide SOM
                deepwide_assigment = assignment_class.make_and_connect(
                    name=f"deepwide_assigment_{algo_}_{assign_name_}",
                    connections=dict(
                        pz_estimate=pz_deepwide_estimator.io.output
                    ),
                )
                self.add_stage(deepwide_assigment)

                # 6.B Find the best cell mapping for all of the wide-field galaxies into the wide SOM
                widewide_assigment = assignment_class.make_and_connect(
                    name=f"widewide_assigment_{algo_}_{assign_name_}",
                    connections=dict(
                        pz_estimate=pz_widewide_estimator.io.output
                    ),
                )
                self.add_stage(widewide_assigment)

                # 8.B Find the best cell mapping for all of the spectroscopic galaxies into the wide SOM
                widespec_assigment = assignment_class.make_and_connect(
                    name=f"widespec_assigment_{algo_}_{assign_name_}",
                    connections=dict(
                        pz_estimate=pz_widespec_estimator.io.output
                    ),
                )
                self.add_stage(widespec_assigment)

                # 4. Use these cell assignments to compute the pz_c redshift histograms in deep SOM.
                # These distributions are redshift pdfs for individual deep SOM cells.
                som_pzc = SOMPZPzc.make_and_connect(
                    name=f"som_pzc_{algo_}_{assign_name_}",
                    redshift_col="redshift",
                    bin_edges=bin_edges_deep,
                    zbins_min=zbins_min_deep,
                    zbins_max=zbins_max_deep,
                    zbins_dz=zbins_dz_deep,
                    deep_groupname="",
                    aliases=dict(spec_data="input_spec_data"),
                    connections=dict(
                        cell_deep_spec_data=deepspec_assigment.io.assignment
                    ),
                )
                self.add_stage(som_pzc)

                # 5. Compute the 'transfer function'.
                # The 'transfer function' weights relating deep to wide photometry.
                # These weights set the relative importance of p(z) from deep SOM cells for each
                # corresponding wide SOM cell.
                # These are traditionally made by injecting galaxies into images with Balrog.
                som_pcchat = SOMPZPc_chat.make_and_connect(
                    name=f"som_pcchat_{algo_}_{assign_name_}",
                    connections=dict(
                        cell_deep_balrog_data=deepdeep_assigment.io.assignment,
                        cell_wide_balrog_data=deepwide_assigment.io.assignment,
                    )
                )
                self.add_stage(som_pcchat)

                # 7. Compute more weights.
                # These weights represent the normalized occupation fraction of each wide SOM cell
                # relative to the full sample.
                som_pzchat = SOMPZPzchat.make_and_connect(
                    name=f"som_pzchat_{algo_}_{assign_name_}",
                    bin_edges=bin_edges_tomo,
                    zbins_min=zbins_min_tomo,
                    zbins_max=zbins_max_tomo,
                    zbins_dz=zbins_dz_tomo,
                    redshift_col="redshift",
                    aliases=dict(
                        spec_data="input_spec_data",
                    ),
                    connections=dict(
                        cell_deep_spec_data=deepspec_assigment.io.assignment,
                        cell_wide_wide_data=widewide_assigment.io.assignment,
                        pz_c=som_pzc.io.pz_c,
                        pc_chat=som_pcchat.io.pc_chat,
                    ),
                )
                self.add_stage(som_pzchat)

                # 9. Define a tomographic bin mapping
                som_tomobin = SOMPZTomobin.make_and_connect(
                    name=f"som_tomobin_{algo_}_{assign_name_}",
                    bin_edges=bin_edges_tomo,
                    zbins_min=zbins_min_tomo,
                    zbins_max=zbins_max_tomo,
                    zbins_dz=zbins_dz_tomo,
                    wide_som_size=300,
                    deep_som_size=300,
                    redshift_col="redshift",
                    aliases=dict(
                        spec_data="input_spec_data",
                    ),
                    connections=dict(
                        cell_deep_spec_data=deepspec_assigment.io.assignment,
                        cell_wide_spec_data=widespec_assigment.io.assignment,
                    ),
                )
                self.add_stage(som_tomobin)

                # 10. Assemble the final tomographic bin estimates
                som_nz = SOMPZnz.make_and_connect(
                    name=f"som_nz_{algo_}_{assign_name_}",
                    bin_edges=bin_edges_tomo,
                    zbins_min=zbins_min_tomo,
                    zbins_max=zbins_max_tomo,
                    zbins_dz=zbins_dz_tomo,
                    redshift_col="redshift",
                    aliases=dict(
                        spec_data="input_spec_data",
                    ),
                    connections=dict(
                        cell_deep_spec_data=deepspec_assigment.io.assignment,
                        cell_wide_wide_data=widewide_assigment.io.assignment,
                        tomo_bins_wide=som_tomobin.io.tomo_bins_wide,
                        pc_chat=som_pcchat.io.pc_chat,
                    ),
                )
                self.add_stage(som_nz)
