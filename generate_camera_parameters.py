import numpy as np
import pandas as pd
import math
from scipy.spatial.transform import Rotation
import random

def generate_camera_parameters_circle(N, R, DEPTH, min_fc, max_fc, pp1, pp2, min_kc, max_kc):
    camera_ids, rxs, rys, rzs, txs, tys, tzs, fc1s, fc2s, pp1s, pp2s, kc1s, kc2s, kc3s, kc4s, kc5s = ([] for _ in range(16))

    for i in range(N):
        theta = 2 * math.pi * i / N  # Angle for the current camera

        # Position of the camera based on radius R and angle theta
        x = R * math.cos(theta)
        y = R * math.sin(theta)
        z = 0

        # Orientation towards the origin
        # Calculate the direction vector from the camera to the origin
        direction_to_origin = np.array([0, 0, DEPTH]) - np.array([x, y, z])
        direction_to_origin = direction_to_origin / np.linalg.norm(direction_to_origin)

        forward_direction = np.array([0,0,-1])
        rotation_vector = np.cross(forward_direction, direction_to_origin)
        rotation_angle = np.arccos(np.clip(np.dot(forward_direction, direction_to_origin), -1.0, 1.0))
        rotation = Rotation.from_rotvec(rotation_angle * rotation_vector)

        euler_angles = rotation.as_euler('xyz', degrees=False)

        # # Calculate yaw and pitch
        # yaw = math.atan2(direction_to_origin[1], direction_to_origin[0])
        # pitch = math.asin(direction_to_origin[2])
        # roll = 0  # Assuming roll is 0 for simplicity

        camera_ids.append(i+1)
        rxs.append(euler_angles[0])
        rys.append(euler_angles[1])
        rzs.append(euler_angles[2])
        txs.append(x)
        tys.append(y)
        tzs.append(z)
        fc = random.randint(min_fc, max_fc)
        fc1s.append(fc)
        fc2s.append(fc)
        pp1s.append(pp1)
        pp2s.append(pp2)
        kc1s.append(random.uniform(min_kc, max_kc))
        kc2s.append(random.uniform(min_kc, max_kc))
        kc3s.append(random.uniform(min_kc, max_kc))
        kc4s.append(random.uniform(min_kc, max_kc))
        kc5s.append(random.uniform(min_kc, max_kc))

    # Create DataFrame
    df = pd.DataFrame({
        'Camera ID': camera_ids,
        'rx': rxs,
        'ry': rys,
        'rz': rzs,
        'tx': txs,
        'ty': tys,
        'tz': tzs,
        'fc1': fc1s,
        'fc2': fc2s,
        'pp1': pp1s,
        'pp2': pp2s,
        'kc1': kc1s,
        'kc2': kc2s,
        'kc3': kc3s,
        'kc4': kc4s,
        'kc5': kc5s
    })

    # Save to CSV
    csv_filename = 'camera_parameters.csv'
    df.to_csv(csv_filename, index=False)
    print(f'Successfully created {csv_filename} with {N} camera parameters.')
    return

def generate_camera_parameters_arc(N, R, DEPTH, ARC_LENGTH, min_fc, max_fc, pp1, pp2, min_kc, max_kc):
    camera_ids, rxs, rys, rzs, txs, tys, tzs, fc1s, fc2s, pp1s, pp2s, kc1s, kc2s, kc3s, kc4s, kc5s = ([] for _ in range(16))

    for i in range(N):
        # Position the camera along an arc which is a part of a circle with radius R
        # ARC_LENGTH controls the span of the arc, a smaller value results in a less curved arc
        arc_angle = ARC_LENGTH * (i / (N - 1) - 0.5)  # Centers the arc and its angle ranges based on ARC_LENGTH
        x = R * math.cos(arc_angle)
        y = R * math.sin(arc_angle)
        z = 0

        # Orientation towards a common focus point along the DEPTH axis
        focus_point = np.array([0, 0, DEPTH])  # All cameras focus towards this point
        camera_position = np.array([x, y, z])
        direction_to_focus = focus_point - camera_position
        direction_to_focus_normalized = direction_to_focus / np.linalg.norm(direction_to_focus)

        forward_direction = np.array([0,0,-1])  # Assuming cameras are initially facing the negative z-axis
        rotation_vector = np.cross(forward_direction, direction_to_focus_normalized)
        rotation_angle = np.arccos(np.clip(np.dot(forward_direction, direction_to_focus_normalized), -1.0, 1.0))
        rotation = Rotation.from_rotvec(rotation_angle * rotation_vector)

        euler_angles = rotation.as_euler('xyz', degrees=False)

        camera_ids.append(i+1)
        rxs.append(euler_angles[0])
        rys.append(euler_angles[1])
        rzs.append(euler_angles[2])
        txs.append(x)
        tys.append(y)
        tzs.append(z)
        fc = random.randint(min_fc, max_fc)
        fc1s.append(fc)
        fc2s.append(fc)
        pp1s.append(pp1)
        pp2s.append(pp2)
        kc1s.append(random.uniform(min_kc, max_kc))
        kc2s.append(random.uniform(min_kc, max_kc))
        kc3s.append(random.uniform(min_kc, max_kc))
        kc4s.append(random.uniform(min_kc, max_kc))
        kc5s.append(random.uniform(min_kc, max_kc))

    # Create DataFrame
    df = pd.DataFrame({
        'Camera ID': camera_ids,
        'rx': rxs,
        'ry': rys,
        'rz': rzs,
        'tx': txs,
        'ty': tys,
        'tz': tzs,
        'fc1': fc1s,
        'fc2': fc2s,
        'pp1': pp1s,
        'pp2': pp2s,
        'kc1': kc1s,
        'kc2': kc2s,
        'kc3': kc3s,
        'kc4': kc4s,
        'kc5': kc5s
    })

    # Save to CSV
    csv_filename = 'camera_parameters.csv'
    df.to_csv(csv_filename, index=False)
    print(f'Successfully created {csv_filename} with {N} camera parameters.')


# Example usage
# generate_camera_parameters_arc(N=10, R=200, DEPTH=0, ARC_LENGTH=math.pi/2, min_fc=1600, max_fc=1650, pp1=1024, pp2=544, min_kc=-1.0, max_kc=1.0)
generate_camera_parameters_circle(N=10, R=200, DEPTH=-800, min_fc=1600, max_fc=1650, pp1=1024, pp2=544, min_kc=-1.0, max_kc=1.0)

