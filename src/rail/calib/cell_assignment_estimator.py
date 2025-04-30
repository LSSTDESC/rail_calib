"""
A PZEstimator that uses a deeper training sample to recalibrate
a p(z) estimate
"""
from typing import Any

import numpy as np

import qp

from ceci.config import StageParameter as Param

from rail.core.model import Model
from rail.core.data import DataHandle, TableLike, Hdf5Handle, ModelHandle
from rail.core.common_params import SHARED_PARAMS
from rail.estimation.informer import PzInformer
from rail.estimation.estimator import PzEstimator
from .cell_assignment_funcs import get_mode_cells, get_max_p_integral_cells


class CellAssignmentPzInformer(PzInformer):
    """Informer that simply bins up p(z) using a point estimate and the
    recalibrates the contents of each bin.
    """

    name = "CellAssignmentPzInformer"
    config_options = PzInformer.config_options.copy()
    config_options.update(
        chunk_size=SHARED_PARAMS,
        hdf5_groupname=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        zmin=Param(float, 0.0, msg="Minimum redshift of the sample"),
        zmax=Param(float, 3.0, msg="Maximum redshift of the sample"),
        nzbins=Param(int, 300, msg="Number of z bins to use in the histogram"),
        ncells=Param(int, 50, msg="Number of cells"),
    )

    outputs = [("model", ModelHandle), ('assignment', Hdf5Handle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        super().__init__(args, **kwargs)
        self.cell_grid: np.ndarray | None=None
        self.z_grid: np.ndarray | None=None
        self.single_hist: np.ndarray | None=None
        self._assignment_handle: DataHandle | None=None

    def _initialize_run(self) -> None:
        self._assignment_handle = None
        self.cell_grid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.ncells + 1
        )
        self.z_grid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.nzbins + 1
        )
        self.single_hist = np.zeros((self.config.ncells+1, self.config.nzbins+1))

    def _finalize_run(self) -> None:
        assert self._assignment_handle is not None
        self._assignment_handle.finalize_write()
        if self.comm is not None:  # pragma: no cover
            self.single_hist = self.comm.reduce(self.single_hist)
        if self.rank == 0:
            assert self.single_hist is not None
            n_total = self.single_hist.sum()
            self.single_hist /= n_total
            model_data = dict(
                n_total=n_total,
                cell_grid=self.cell_grid,
                z_grid=self.z_grid,
                hist=self.single_hist,
            )
            model = Model(
                model_data,
                creation_class_name=self.__class__.__name__,
                version=0,
                provenance=dict(),
            )
            self.add_data("model", model)

    def _process_chunk(
        self,
        start: int,
        end: int,
        data: qp.Ensemble,
        true_redshift: TableLike,
        first: bool,
    ) -> None:
        assignment_data = self._get_cells_and_dist(data)
        self._do_chunk_output(assignment_data, start, end, first)
        cells = assignment_data['cells']
        assert self.z_grid is not None
        assert self.cell_grid is not None
        assert self.single_hist is not None
        true_bins = np.squeeze(np.searchsorted(self.z_grid, true_redshift, side='left', sorter=None))
        # do something faster with numpy??
        for i, j in zip(cells, true_bins):
            self.single_hist[i, min(j, self.config.nzbins)] += 1

    def _get_cells_and_dist(self, data: qp.Ensemble) -> TableLike:
        raise NotImplementedError()

    def _do_chunk_output(
        self, out_data: TableLike, start: int, end: int, first: bool
    ) -> None:
        if first:
            the_handle = self.add_handle("assignment", data=out_data)
            assert isinstance(the_handle, Hdf5Handle)
            self._assignment_handle = the_handle
            if self.config.output_mode != "return":
                self._assignment_handle.initialize_write(
                    self._input_length, communicator=self.comm
                )
        assert self._assignment_handle is not None
        self._assignment_handle.set_data(out_data, partial=True)
        if self.config.output_mode != "return":
            self._assignment_handle.write_chunk(start, end)
        return out_data

    def run(self) -> None:
        iterator = self._setup_iterator()
        self._initialize_run()

        first = True
        for s, e, qp_ens, true_redshift in iterator:
            print(f"Process {self.rank} running estimator on chunk {s} - {e}")
            self._process_chunk(s, e, qp_ens, true_redshift, first)
            first = False

        self._finalize_run()
    

class PZModeCellAssignmentPzInformer(CellAssignmentPzInformer):
    """Stage that assigns object to cells on mode of p(z) estimate
    """

    name = "PZModeCellAssignmentPzInformer"
    config_options = CellAssignmentPzInformer.config_options.copy()

    def _get_cells_and_dist(self, data: qp.Ensemble) -> TableLike:
        assert self.cell_grid is not None
        cells = get_mode_cells(data, self.cell_grid)
        dist = np.ones(cells.shape)
        return dict(
            cells=cells,
            dist=dist,
        )


class PZMaxCellPCellAssignmentPzInformer(CellAssignmentPzInformer):
    """Stage that assigns object to cells based the cell with
    the highest integrated p(z)
    """

    name = "PZMaxCellPCellAssignmentPzInformer"
    config_options = CellAssignmentPzInformer.config_options.copy()

    def _get_cells_and_dist(self, data: qp.Ensemble) -> TableLike:
        assert self.cell_grid is not None
        cells = get_max_p_integral_cells(data, self.cell_grid)
        dist = np.ones(cells.shape)
        return dict(
            cells=cells,
            dist=dist,
        )


class CellAssignmentPzEstimator(PzEstimator):
    """Estimator which takes an existing p(z) estimate and
    recalibrates it using a model trained on a more representative
    data set.
    """

    name = "CellAssignmentPzEstimator"
    config_options = PzEstimator.config_options.copy()

    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, first: bool
    ) -> None:

        z_grid=self.model.data['z_grid']
        cell_grid=self.model.data['cell_grid']
        hist=self.model.data['hist']

        cells = self._get_cells(data, cell_grid)
        recalib_vals = hist[cells][:,:-1]
        overflow = hist[cells][:,-1]

        qp_d = qp.Ensemble(
            qp.hist,
            data=dict(bins=z_grid, pdfs=recalib_vals),
        )
        qp_d.set_ancil(
            dict(
                cells=cells,
                zmedian=np.median(recalib_vals, axis=1),
                overflow=overflow,
            )
        )
        self._do_chunk_output(qp_d, start, end, first)

    def _get_cells(self, data: qp.Ensemble, cell_grid: np.ndarray) -> TableLike:
        raise NotImplementedError()


class PZModeCellAssignmentPzEstimator(CellAssignmentPzEstimator):
    """Stage that assigns object to cells on mode of p(z) estimate
    """

    name = "PZModeCellAssignmentPzEstimator"
    config_options = CellAssignmentPzEstimator.config_options.copy()

    def _get_cells(self, data: qp.Ensemble, cell_grid: np.ndarray) -> np.ndarray:
        return get_mode_cells(data, cell_grid)


class PZMaxCellPCellAssignmentPzEstimator(CellAssignmentPzEstimator):
    """Stage that assigns object to cells based the cell with
    the highest integrated p(z)
    """

    name = "PZMaxCellPCellAssignmentPzEsimator"
    config_options = CellAssignmentPzEstimator.config_options.copy()

    def _get_cells(self, data: qp.Ensemble, cell_grid: np.ndarray) -> np.ndarray:
        return get_max_p_integral_cells(data, cell_grid)
