# This code was originally written by Simon Alexanderson (github.com/simonalexanderson),
# but the current version is taken from the following commit by Taras Kucherenko: 
# https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder/commit/daffa184779fb45c5a6ff1279a01c5e85b0eda37#diff-d95865ebf5428aa201840818a88d0fb5
"""
This script converts a gesticulation dataset from the BVH format to joint angles.
Input: BVH files
Output: .npz of each file containing the rotation angles.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from argparse import ArgumentParser

import glob
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from visualization.pymo.parsers import BVHParser
from visualization.pymo.data import Joint, MocapData
from visualization.pymo.preprocessing import *
from visualization.pymo.writers import *

import joblib as jl

def extract_joint_angles(bvh_dir, files, dest_dir, pipeline_dir, fps):
    p = BVHParser()

    if not os.path.exists(pipeline_dir):
        raise Exception("Pipeline dir for the motion processing ", pipeline_dir, " does not exist! Change -pipe flag value.")

    data_all = list()
    for f in files:
        ff = os.path.join(bvh_dir, f + '.bvh')
        print(ff)
        data_all.append(p.parse(ff))

    data_pipe = Pipeline([
        ## https://github.com/omimo/PyMO
        ## https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
       ('dwnsampl', DownSampler(tgt_fps=fps,  keep_all=False)),
       ('root', RootTransformer('hip_centric')),
       ('mir', Mirror(axis='X', append=True)),
       ('jtsel', JointSelector(['Spine','Spine1','Spine2','Spine3','Neck','Neck1','Head','RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'], include_root=True)),
       ('exp', MocapParameterizer('expmap')), ## Conversion of the motion capture data into exponential maps, keeping the rotation
       ('cnst', ConstantsRemover()),
       ('np', Numpyfier()) ## Convert the info into numpy array
    ])

    out_data = data_pipe.fit_transform(data_all)
    
    # the datapipe will append the mirrored files to the end
    assert len(out_data) == 2*len(files) ## Here we check that we have the mirrored data + the original data as "our data"
    
    jl.dump(data_pipe, os.path.join(pipeline_dir + 'data_pipe.sav')) ## Save the pipeline
        
    fi=0
    for f in files:
        ff = os.path.join(dest_dir, f)
        print(ff)
        np.savez(ff + ".npz", clips=out_data[fi]) ## Save data as npz
        np.savez(ff + "_mirrored.npz", clips=out_data[len(files)+fi]) ## Save mirrored data as npz
        fi=fi+1



if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser()
    parser.add_argument('--bvh_dir', '-orig', default="/data/dataset_trinity/raw_data/Motion", # Changed for the redsofa computer
                                   help="Path where original motion files (in BVH format) are stored")
    parser.add_argument('--dest_dir', '-dest', default="/data/dataset_trinity/raw_data/Motion", # Changed for the redsofa computer
                                   help="Path where extracted motion features will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="../utils/",
                        help="Path where the motion data processing pipeline will be stored")

    params = parser.parse_args()

    files = []
    # Go over all BVH files
    print("Going to pre-process the following motion files:")
    for r, d, f in os.walk(params.bvh_dir): ## r = root, d = directory, f = file
        for file in f:
            print(file)
            if '.bvh' in file:
                ff=os.path.join(r, file)
                basename = os.path.splitext(os.path.basename(ff))[0] ## Extract the basename without the extension.
                files.append(basename)

    extract_joint_angles(params.bvh_dir, files, params.dest_dir, params.pipeline_dir , fps=60)
    print("FINISH!")
