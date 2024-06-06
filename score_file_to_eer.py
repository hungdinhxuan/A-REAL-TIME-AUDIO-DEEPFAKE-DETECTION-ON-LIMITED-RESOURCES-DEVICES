import sys
import os.path
import numpy as np
import pandas
import eval_metrics_DF as em


def eval_to_score_file(score_file, cm_key_file, phase):
    # CM key file is the metadata file that contains the ground truth labels for the eval set
    # score file is the output of the system that contains the scores for the eval set
    # phase is the phase of the eval set (dev or eval)

    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(
        score_file, sep=' ', header=None, skipinitialspace=True)
    # check here for progress vs eval set
    cm_scores = submission_scores.merge(
        cm_data[cm_data[1] == phase] if phase != 'all'  else cm_data, left_on=0, right_on=0, how='inner')
    # cm_scores.head()
    #  0       1_x   1_y      2      3
    #  a.wav  1.234   eval   Music   spoof
    bona_cm = cm_scores[cm_scores[3] == 'bonafide']['1_x'].values
    spoof_cm = cm_scores[cm_scores[3] == 'spoof']['1_x'].values

    max_score = max(cm_scores['1_x'].values)
    min_score = min(cm_scores['1_x'].values)

    print("max score: ", max_score)
    print("min score: ", min_score)

    eer_cm, th = em.compute_eer(bona_cm, spoof_cm)
    out_data = "eer: {}\tthreshold: {}\n".format(100*eer_cm, th)
    print(out_data)
    return eer_cm


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("CHECK: invalid input arguments. Please read the instruction below:")
        print(__doc__)
        exit(1)

    submit_file = sys.argv[1]
    cm_key_file = sys.argv[2]
    phase = sys.argv[3]

    if not os.path.isfile(submit_file):
        print("%s doesn't exist" % (submit_file))
        exit(1)

    if not os.path.isfile(cm_key_file):
        print("%s doesn't exist" % (cm_key_file))
        exit(1)

    _ = eval_to_score_file(submit_file, cm_key_file, phase)
