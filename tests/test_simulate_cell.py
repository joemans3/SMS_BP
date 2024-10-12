import numpy as np
import pytest

from SMS_BP.simulate_cell import make_directory_structure, sub_segment


# Test for sub_segment function
def test_sub_segment():
    img = np.random.rand(3, 10, 10)  # 3 frames of 10x10
    subsegment_num = 3

    # Test mean subsegment
    result = sub_segment(img, subsegment_num, subsegment_type="mean")
    assert isinstance(result, list)
    assert len(result) == subsegment_num
    assert result[0].shape == (
        10,
        10,
    )  # Each subsegment should have the same image shape

    # Test invalid subsegment type
    with pytest.raises(ValueError):
        sub_segment(img, subsegment_num, subsegment_type="invalid")


# Test for make_directory_structure function
def test_make_directory_structure(tmpdir):
    cd = "test_dir"
    img_name = "test_image"
    img = np.random.rand(10, 10)
    subsegment_type = "mean"
    subsegment_num = 2

    # Test the function
    make_directory_structure(tmpdir, img_name, img, subsegment_type, subsegment_num)
    tmpdir.remove(cd)
