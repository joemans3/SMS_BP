import os
from unittest import mock

import numpy as np
import pytest

from SMS_BP.simulate_cell import make_directory_structure, save_tiff, sub_segment


# Test for save_tiff function
def test_save_tiff(tmpdir):
    image = np.random.rand(10, 10)
    path = tmpdir.mkdir("test_dir")
    img_name = "test_image"

    # Test saving with a provided image name
    save_tiff(image, str(path), img_name)
    assert os.path.exists(os.path.join(path, img_name + ".tiff"))

    # Test saving without an image name
    save_tiff(image, str(path))
    assert os.path.exists(os.path.join(path, "image.tiff"))


# Test for sub_segment function
def test_sub_segment():
    img = np.random.rand(10, 10, 3)  # 3 frames of 10x10
    sub_frame_num = 3

    # Test mean subsegment
    result = sub_segment(img, sub_frame_num, subsegment_type="mean")
    assert isinstance(result, list)
    assert len(result) == sub_frame_num
    assert result[0].shape == (
        10,
        10,
    )  # Each subsegment should have the same image shape

    # Test invalid subsegment type
    with pytest.raises(ValueError):
        sub_segment(img, sub_frame_num, subsegment_type="invalid")


# Test for make_directory_structure function
@mock.patch("os.makedirs")
def test_make_directory_structure(mock_makedirs):
    # Mock directory structure and call the function
    cd = "test_dir"
    img_name = "test_image"
    img = np.random.rand(10, 10)
    subsegment_type = "mean"
    sub_frame_num = 2

    # Test the function
    make_directory_structure(cd, img_name, img, subsegment_type, sub_frame_num)

    # Check that the directories were made
    mock_makedirs.assert_called()
