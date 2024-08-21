# Neural Recalibration for Multi-Camera Systems
Official repo of "Neural Real-Time Recalibration for Infrared Multi-Camera Systems".

We provide two variants: one for point-based calibration (`neural_recalib_point.py`) and another for image-based calibration (`neural_recalib_image.py`). The calibration process involves training a neural model specific to a pair of calibration object and camera configuration.

## Contents

### Directories
- **calib_objects/**: Contains example geometry files for different calibration objects.
  - `fiducials_cube_corners.csv`
  - `fiducials_sphere_64_new.csv`
  - `fiducials_sphere_64.csv`

- **OEM_initial_calib/**: Contains example OEM initial calibrations for different multi-camera configurations.
  - `camera_parameters_10.csv`
  - `camera_parameters.csv`
  - `camera_params_U_12348910.csv`

- **logs/**: Directory where logs of training and testing runs are saved.

- **output/**: Directory where the outputs of the recalibration process are saved.

### Files
  - **camera_pose_synthesis.py**
  - **diffproj.py**
  - **draw_utilities.py**
  - **generate_camera_parameters.py**
  - **networks.py**
  - **neural_recalib_image.py**
  - **neural_recalib_point.py**
  - **utils.py**

- **README.md**: This file.
- **LICENSE**: Licensing information for this work.

## Usage

To perform training or testing, use the following commands:

- For point-based calibration:
```
python3 neural_recalib_point.py --mode train --load_last_best n
```

- For image-based calibration:
```
python3 neural_recalib_image.py --mode train --load_last_best n
```

## Training Progression

For an overview of how the training progresses, see the animation below:

![Training Progression](training_progression.gif)

## Dependencies

To run the code, you need to install the following dependencies. Please provide the list of dependencies, and I will include them here.


---

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. See [LICENSE](LICENSE) for more information.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
