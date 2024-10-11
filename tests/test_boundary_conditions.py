from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import Optional

from SMS_BP.boundary_conditions import _absorbing_boundary, _refecting_boundary
from SMS_BP.errors import HurstHighError


# setup dataclass for testing boundary boundary_conditions
@dataclass
class BoundaryConditions:
    fbm_store_last: float
    fbm_candidate: float
    space_lim: np.ndarray
    expected_result: float
    test_ID: str
    expected_error: Optional[type[Exception]] = None
    expected_error_message: Optional[str] = None


test_conditions_reflecting_boundary = [
    BoundaryConditions(
        fbm_store_last=1.0,
        fbm_candidate=0.0,
        space_lim=np.array([0, 1]),
        expected_result=0.0,
        test_ID="Test _refecting_boundary with candidate at boundary",
    ),
    BoundaryConditions(
        fbm_store_last=0.0,
        fbm_candidate=1.0,
        space_lim=np.array([0, 1]),
        expected_result=1.0,
        test_ID="Test _refecting_boundary with fbm_store_last at boundary and fbm_candidate inside",
    ),
    BoundaryConditions(
        fbm_store_last=0.5,
        fbm_candidate=2.0,
        space_lim=np.array([0, 1]),
        expected_result=0.0,
        test_ID="Test _refecting_boundary with fbm_store_last inside and fbm_candidate outside positive",
    ),
    BoundaryConditions(
        fbm_store_last=0.5,
        fbm_candidate=-1.0,
        space_lim=np.array([0, 1]),
        expected_result=1.0,
        test_ID="Test _refecting_boundary with fbm_store_last inside and fbm_candidate outside negative",
    ),
    BoundaryConditions(
        fbm_store_last=0.5,
        fbm_candidate=0.3,
        space_lim=np.array([-1, 1]),
        expected_result=0.3,
        test_ID="Test negative space_lim with fbm_store_last inside and fbm_candidate inside",
    ),
    # test recursion error
    BoundaryConditions(
        fbm_store_last=0.5,
        fbm_candidate=1e10,  # will force recursion error with max depth 1000
        space_lim=np.array([-1, 1]),
        expected_result=0.3,
        test_ID="Test recursion error throw with 1e10 candidate and -1,1 space limit",
        expected_error=HurstHighError,
        expected_error_message="You are probably using H > 0.5 in a small space limit. Try to increase the space limit or decrease the H value. \n Since H > 0.5, it will compound the step sizes.",
    ),
]


@pytest.mark.parametrize(
    "boundary_conditions", test_conditions_reflecting_boundary, ids=lambda x: x.test_ID
)
def test_reflecting_boundary(boundary_conditions: BoundaryConditions):
    if boundary_conditions.expected_error is not None:
        with pytest.raises(boundary_conditions.expected_error) as error_info:
            _refecting_boundary(
                boundary_conditions.fbm_store_last,
                boundary_conditions.fbm_candidate,
                boundary_conditions.space_lim,
            )
        assert str(error_info.value) == boundary_conditions.expected_error_message
    else:
        result = _refecting_boundary(
            boundary_conditions.fbm_store_last,
            boundary_conditions.fbm_candidate,
            boundary_conditions.space_lim,
        )
        assert result == boundary_conditions.expected_result


test_conditions_absorbing_boundary = [
    BoundaryConditions(
        fbm_store_last=1.0,
        fbm_candidate=0.0,
        space_lim=np.array([0, 1]),
        expected_result=0.0,
        test_ID="Test _absorbing_boundary with candidate at boundary",
    ),
    BoundaryConditions(
        fbm_store_last=0.0,
        fbm_candidate=1.0,
        space_lim=np.array([0, 1]),
        expected_result=1.0,
        test_ID="Test _absorbing_boundary with fbm_store_last at boundary and fbm_candidate inside",
    ),
    BoundaryConditions(
        fbm_store_last=0.5,
        fbm_candidate=2.0,
        space_lim=np.array([0, 1]),
        expected_result=1.0,
        test_ID="Test _absorbing_boundary with fbm_store_last inside and fbm_candidate outside positive",
    ),
    BoundaryConditions(
        fbm_store_last=0.5,
        fbm_candidate=-1.0,
        space_lim=np.array([0, 1]),
        expected_result=0.0,
        test_ID="Test _absorbing_boundary with fbm_store_last inside and fbm_candidate outside negative",
    ),
    BoundaryConditions(
        fbm_store_last=0.5,
        fbm_candidate=-1.3,
        space_lim=np.array([-1, 1]),
        expected_result=-1.0,
        test_ID="Test _absorbing_boundary negative space_lim with fbm_store_last inside and fbm_candidate outside negative",
    ),
]


@pytest.mark.parametrize(
    "boundary_conditions", test_conditions_absorbing_boundary, ids=lambda x: x.test_ID
)
def test_absorbing_boundary(boundary_conditions: BoundaryConditions):
    result = _absorbing_boundary(
        boundary_conditions.fbm_store_last,
        boundary_conditions.fbm_candidate,
        boundary_conditions.space_lim,
    )
    assert result == boundary_conditions.expected_result
