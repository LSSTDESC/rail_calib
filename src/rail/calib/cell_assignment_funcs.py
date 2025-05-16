import numpy as np
import qp


def get_mode_cells(data: qp.Ensemble, cell_grid: np.ndarray) -> np.ndarray:
    """ Get the cell that contains the mode of the p(z) distribution

    Parameters
    ----------
    data:
        pdfs, should contain 'zmode', in the ancillary data

    cell_grid:
        cell bin edges

    Returns
    -------
    cell assignments
    """
    try:
        point_estimates = data.ancil['zmode']
    except:
        tight_grid = np.linspace(cell_grid[0], cell_grid[1], 2*len(cell_grid) - 1)
        tight_grid = 0.5 * (tight_grid[0:-1] + tight_grid[1:])
        pdfs = data.pdf(tight_grid)
        point_estimates = tight_grid[np.argmax(pdfs, axis=1)]
    cells = np.squeeze(np.searchsorted(cell_grid, point_estimates, side='left', sorter=None))
    n_cells = len(cell_grid) - 1
    cells = np.where(cells>=n_cells, n_cells-1, cells)
    return cells


def get_max_p_integral_cells(data: qp.Ensemble, cell_grid: np.ndarray) -> np.ndarray:
    """ Get the cell with the highest integrated p(z)

    Parameters
    ----------
    data:
        pdfs

    cell_grid:
        cell bin edges

    Returns
    -------
    cell assignments
    """
    cdfs = data.cdf(cell_grid)
    cell_pdf_integrals = cdfs[:,1:] - cdfs[:,0:-1]
    cells = np.squeeze(np.argmax(cell_pdf_integrals, axis=1))
    return cells
