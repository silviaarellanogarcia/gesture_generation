import sys
import librosa
import librosa.display
import numpy as np

## BEAT EXTRACTION CODE 
# Using the Librosa Library: https://librosa.org/doc/0.10.1/generated/librosa.beat.beat_track.html

def beat_extraction(audio_file_path=''):
    """
    Compute the beat timestamps of an audio
    """
    # Load audio file
    if audio_file_path == '':
        print("ERROR: The audio file has not been specified!!")
        return
    
    y, sr = librosa.load(audio_file_path, duration=15.0) # y represents the audio samples and sr is the sampling rate

    # Calculate tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Convert beat frames to time
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return beat_times

## KINEMATIC BEAT EXTRACTION

def count_indentation(line):
    """
    Count the number of leading tabs or spaces in a line.
    """
    count = 0
    for char in line:
        if char == '\t':  # Check for tabs
            count += 1
        elif char == ' ':  # Check for spaces
            count += 1
        else:
            break  # Stop counting when encountering a non-tab/non-space character
    return count

def find_tuple_with_higher_hierarchy(my_hierarchy, order_list):
    """
    Finds the parent bone of a specific sequence of connected bones
    """
    for item in reversed(order_list):
        if item[1] < my_hierarchy:
            return item
    return None

def parse_hierarchy(lines):
    """
      Parse bone hierarchy from BVH file.

      Input:
      - lines: List of lines from the BVH file.

      Output:
      - Updated index on the lines list.
    """
    bone_hierarchies = []
    current_hierarchy = []
    order = []

    for i in range(len(lines)):
        n_indentations = count_indentation(lines[i])
        line = lines[i].strip()
        if (line.startswith('JOINT') or line.startswith('ROOT')):
            bone_name = line.split(' ')[1]
            if(line.startswith('ROOT')):
                current_hierarchy = []
            elif(n_indentations < order[-1][1]):
                new_sub_root = find_tuple_with_higher_hierarchy(n_indentations, order)
                if(current_hierarchy != []):
                    bone_hierarchies.append(current_hierarchy)
                    current_hierarchy = [new_sub_root[0]]
            current_hierarchy.append(bone_name)
            order.append((bone_name, n_indentations))

        if(line.startswith('MOTION')):
            index = i

    bone_hierarchies.append(current_hierarchy)

    return bone_hierarchies, order, index

def parse_motion(lines, index):
    """
    Parse motion data from BVH file.

    Args:
    - lines: List of lines from the BVH file.
    - index: Current index in the lines list.
    - frames: List to store joint angles for each frame.

    Returns:
    - Updated frames list.
    """
    list_angles_per_frame = []
    frames = []
    while index < len(lines):
        line = lines[index].strip()
        if line == 'MOTION' or line.startswith('Frame Time:'):
            pass
        elif line.startswith('Frames:'):
            n_frames = int(line.split()[-1])
        else:
            list_angles_per_frame = []
            angles = list(map(float, line.split()))

            counter = 3
            counter_bones = 0
            while(counter < len(angles)):
                ang = angles[counter : counter + 3]
                list_angles_per_frame.append(ang)
                counter += 3
                counter_bones += 1

            frames.append(list_angles_per_frame)

        index += 1

    return frames

def parse_bvh_file(file_path):
    """
    Parse a BVH file and extract bone hierarchy and joint angles.

    Args:
    - file_path: Path to the BVH file.

    Returns:
    - bone_hierarchy: Dictionary representing bone hierarchy.
    - frames: List of dictionaries containing joint angles for each frame.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    bone_hierarchy, order, index = parse_hierarchy(lines)
    frames = parse_motion(lines, index)

    return bone_hierarchy, frames, order, index

def ComputeAngleChangeRate(frames):
    # Equation (10) to compute the Mean Absolute Angle Change
    angle_change_in_clip = np.zeros(len(frames[0]))

    for t in range(len(frames) - 1):
        dif = np.abs(np.array(frames[t + 1]) - np.array(frames[t]))
        angle_change_in_clip += np.sum(dif, axis=1)

    MAAC = angle_change_in_clip / (len(frames) - 1)

    # angle change rate of each frame
    angle_change_rate = []
    for t in range(len(frames) - 1):
        numerator = np.sum(np.abs((np.array(frames[t + 1]) - np.array(frames[t]))), axis=1)
        denominator = MAAC

        denominator[numerator == 0] = 1 # To avoid divisions by 0. If the denominator is 0, the numerator will be 0 too for sure.

        angle_change_rate.append(np.mean(numerator/denominator))

    return angle_change_rate

if __name__ == "__main__":
    if len(sys.argv) < 1:
        print(
            "usage: python beat_consistency.py <path to BVH file>"
        )
        sys.exit(0)

    bvh_file_path = sys.argv[1]
    frames = parse_bvh_file(bvh_file_path)[1]
    angle_change_rate = ComputeAngleChangeRate(frames)

    ## TODO: Compute the threshold for obtaining the local optima of the movements and compute the Beat Consistency Score (eq.11)
    print(angle_change_rate)