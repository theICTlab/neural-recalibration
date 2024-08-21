# Neural Real-Time Recalibration for Infrared Multi-Camera Systems
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
  - **environment.yml**
  - **generate_camera_parameters.py**
  - **networks.py**
  - **neural_recalib_image.py**
  - **neural_recalib_point.py**
  - **utils.py**

- **README.md**: This file.
- **LICENSE**: Licensing information for this work.

## Getting Started

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/theICTlab/neural-recalibration.git
cd neural-recalibration
```

### 2. Create and Activate a Conda Environment

```bash
conda env create -f environment.yml
conda activate neural-recalibration
```

## Usage

To perform training or testing, use the following command:

```bash
python3 [neural_recalib_point.py | neural_recalib_image.py] --mode [train | test] --load_last_best [y | n]
```
- `[neural_recalib_point.py | neural_recalib_image.py]`: The chosen neural recalibration variant. *Note: we only provide sample data for the point-based variant.*
- `--mode [train | test]`: Specifies whether to train a new model or test an existing one.
- `--load_last_best [y | n]`: Specifies whether to load the last best model (y) or start fresh (n).

### Examples

- For point-based calibration training:
```bash
python3 neural_recalib_point.py --mode train --load_last_best n
```
This command starts training a new model using point-based calibration without loading any previously saved model.

- For image-based calibration testing:
```bash
python3 neural_recalib_image.py --mode test --load_last_best y
```
This command tests an image-based calibration model using the last best-trained model.

## Training Progression

For an overview of how the training progresses, see the animation below:

![Training Progression](example_training/training_progression.gif)

The red circles are the predicted camera poses, and the blue circles are the ground truth camera poses. The yellow indicates the first camera in each multi-camera setup.

---

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. See [LICENSE](LICENSE) for more information.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
