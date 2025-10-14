import numpy as np
from pathlib import Path
from robotdatapy.data.img_data import ImgData
import unittest

class TestImgData(unittest.TestCase):

    def test_from_npy(self):

        # =================== RGB Data ===================
        # Create the class from dummy data
        test_dir = Path(__file__).absolute().parent / 'files' / 'test_img_data' / 'test_from_npy'
        rgb_file_path = test_dir / 'rgb' / 'imgs.npy'
        times_path = test_dir / 'rgb' / 'times.npy'
        img_data = ImgData.from_npy(path=rgb_file_path,
                         path_times=times_path,
                         K=[120, 0, 60, 0, 140, 30, 0,  0,  1],
                         D=[1, 2, 3, 4],
                         width=1000,
                         height=500,
                         encoding='rgb8',
                         time_tol=27,
                         causal=True)
        
        # Load the image file manually
        test_img = np.load(rgb_file_path)[3][..., ::-1]
        test_time = np.load(times_path)[3]

        # Make sure everything lines up as expected
        np.testing.assert_array_equal(img_data._load_img(3), test_img)
        np.testing.assert_equal(img_data.times[3], test_time)
        np.testing.assert_equal(img_data.width, 1000)
        np.testing.assert_equal(img_data.height, 500)
        np.testing.assert_array_equal(img_data.K, np.array([[120, 0, 60], 
                                                            [0, 140, 30], 
                                                            [0,  0,  1]]))
        np.testing.assert_array_equal(img_data.D, np.array([1, 2, 3, 4]))
        np.testing.assert_equal(img_data.time_tol, 27)
        np.testing.assert_equal(img_data.causal, True)

        # =================== Depth Data (32FC1) ===================
        # Create the class from dummy data
        test_dir = Path(__file__).absolute().parent / 'files' / 'test_img_data' / 'test_from_npy'
        depth_file_path = test_dir / 'depth' / 'imgs.npy'
        times_path = test_dir / 'depth' / 'times.npy'
        depth_data = ImgData.from_npy(path=depth_file_path,
                         path_times=times_path,
                         K=[128, 0, 60, 0, 140, 30, 0,  0,  1],
                         D=[5, 1, 2, 3, 4],
                         width=1700,
                         height=300,
                         encoding='32FC1',
                         time_tol=20,
                         causal=False)
        
        # Load the image file manually
        test_img = np.load(depth_file_path)[7]
        test_time = np.load(times_path)[7]

        # Make sure everything lines up as expected
        np.testing.assert_array_equal(depth_data._load_img(7), test_img)
        np.testing.assert_equal(depth_data.times[7], test_time)
        np.testing.assert_equal(depth_data.width, 1700)
        np.testing.assert_equal(depth_data.height, 300)
        np.testing.assert_array_equal(depth_data.K, np.array([[128, 0, 60], 
                                                            [0, 140, 30], 
                                                            [0,  0,  1]]))
        np.testing.assert_array_equal(depth_data.D, np.array([5, 1, 2, 3, 4]))
        np.testing.assert_equal(depth_data.time_tol, 20)
        np.testing.assert_equal(depth_data.causal, False)


if __name__ == '__main__':
    unittest.main()