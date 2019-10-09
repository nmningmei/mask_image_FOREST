
import os
from time import sleep
from glob import glob
import numpy as np

working_dir = ''

timing = 30
bashes = np.sort(glob(os.path.join(working_dir,"*","*q")))
for each in bashes[1:]:
    directory_to_go = os.path.join('/bcbl/home/public/Consciousness/uncon_feat/scripts/MRI/standard/',
                                   each.split('/')[0])
    os.chdir(directory_to_go)
    os.system("qsub {}".format(each.split('/')[-1]))
    sleep(timing)
