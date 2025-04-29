

class PZCellAssigner(RailStage):
    """Base class for stage that assigns object to cells based on p(z) estimate
    """

    name = "PZCellAssignmentBase"
    config_options = RailStage.config_options.copy()
    config_options.update(
        zmin=Param(float, 0.0, msg="Minimum redshift of the sample"),
        zmax=Param(float, 3.0, msg="Maximum redshift of the sample"),        
        ncells=Param(int, 300, msg="Number of cells"),
    )
    inputs = [("pz_estimate", QPHandle)]
    outputs = [('assignment', Hdf5Handle)]

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """Initialize Stage"""
        super().__init__(args, **kwargs)
        self.zgrid: np.ndarray | None = None
        
    def run(self) -> None:
        
        self.zgrid = np.linspace(
            self.config.zmin, self.config.zmax, self.config.ncells + 1
        )
        assert self.zgrid is not None
        first = True
        self._initialize_run()
        self._output_handle = None
        iterator = self.input_iterator("pz_estimate")        
        for s, e, data in iterator:
            print(f"Process {self.rank} running cell assignment on chunk {s} - {e}")
            self._process_chunk(s, e, data)
            first = False
        first = True
        self._finalize_run()

    def _initialize_run(self) -> None:
        self._output_handle = None

    def _finalize_run(self) -> None:
        assert self._output_handle is not None
        self._output_handle.finalize_write()
        
    def _process_chunk(
        self, start: int, end: int, data: qp.Ensemble, first: bool
    ) -> None:
        out_data = self._get_cells_and_dist(data)
        self._do_chunk_output(out_data, start, end, first)

    def _do_chunk_output(
        self, out_data: TableLike, start: int, end: int, first: bool
    ) -> None:
        if first:
            the_handle = self.add_handle("assignment", data=out_data)
            assert isinstance(the_handle, Hdf5Handle)
            self._output_handle = the_handle
            if self.config.output_mode != "return":
                self._output_handle.initialize_write(
                    self._input_length, communicator=self.comm
                )
        assert self._output_handle is not None
        self._output_handle.set_data(out_data, partial=True)
        if self.config.output_mode != "return":
            self._output_handle.write_chunk(start, end)
        return out_data
    
    def _get_cells_and_dist(self, data: qp.Ensemble) -> TableLike:
        raise NotImplementedError()


    
class PZModeCellAssigner(PZCellAssigner):
    """Stage that assigns object to cells on mode of p(z) estimate
    """

    name = "PZModeCellAssigner"
    config_options = PZCellAssigner.config_options.copy()

    def _get_cells_and_dist(self, data: qp.Ensemble) -> TableLike:
        point_estimates = data.ancil['zmode']
        cells = np.squeeze(np.searchsorted(self.zgrid, point_estimates, side='left', sorter=None))
        dist = np.ones((len(point_estimates)))
        return dict(
            cells=cells,
            dist=dist,
        )


class PZMaxCellPCellAssigner(PZCellAssigner):
    """Stage that assigns object to cells based the cell with
    the highest integrated p(z) 
    """

    name = "PZMaxCellPCellAssigner"
    config_options = PZCellAssigner.config_options.copy()

    def _get_cells_and_dist(self, data: qp.Ensemble) -> TableLike:
        pdfs = data.pdf(self.zgrid)        
        cells = np.squeeze(np.argmax(self.zgrid, pdfs, axis=1))                               
        dist = np.ones((data.npdf)))
        return dict(
            cells=cells,
            dist=dist,
        )
    


