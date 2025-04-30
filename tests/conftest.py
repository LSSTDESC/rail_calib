import os
import urllib.request
import pytest


def remove_data() -> None:

    if os.environ.get('NO_TEARDOWN', 0):
        return
    
    try:
        os.unlink("tests/dc2_run2p2i_dr6_test_dered.hdf5")
    except:
        pass
    try:
        os.unlink("tests/output_estimate_knn.hdf5")
    except:
        pass
    

@pytest.fixture(name="get_data", scope="package")
def get_data(request: pytest.FixtureRequest) -> int:

    if not os.path.exists("tests/dc2_run2p2i_dr6_test_dered.hdf5"):
        urllib.request.urlretrieve(
            "http://s3df.slac.stanford.edu/people/echarles/xfer/dc2_run2p2i_dr6_test_dered.hdf5",
            "tests/dc2_run2p2i_dr6_test_dered.hdf5",
        )
        if not os.path.exists("tests/dc2_run2p2i_dr6_test_dered.hdf5"):
            return 1

    if not os.path.exists("tests/output_estimate_knn.hdf5"):
        urllib.request.urlretrieve(
            "http://s3df.slac.stanford.edu/people/echarles/xfer/output_estimate_knn.hdf5",
            "tests/output_estimate_knn.hdf5",
        )
        if not os.path.exists("tests/output_estimate_knn.hdf5"):
            return 2

    request.addfinalizer(remove_data)
    return 0
