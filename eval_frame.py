import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from eval_utils import interpolated_prec_rec
from eval_utils import check_frame, distance_2_center


class FrameDetection(object):

    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_data=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 drange=np.linspace(0.25, 1, 4),
                 subset='validation', verbose=False,
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        self.subset = subset
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        self.drange = drange
        # Retrieve blocked videos from server.
        self.blocked_videos = list()
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_data)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed distance to center ratio: {}'.format(self.drange))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.
        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.
        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        #  if not all([field in data.keys() for field in self.gt_fields]):
            #  raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        return ground_truth, activity_index

    def _import_prediction(self, data):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.
        Parameters
        ----------
        data: json obj 
        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        #  with open(prediction_filename, 'r') as fobj:
        #  data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, frame_lst = [], []
        label_lst, score_lst = [], []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            for result in v:
                #  import pdb;pdb.set_trace()
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                frame_lst.append(float(result['frame']))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   'frame': frame_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label. 
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' %
                  label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.drange), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(
                    cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx), drange=self.drange,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print('\t mAP: {}'.format(
                ','.join(map(lambda x: str(round(x, 5)), self.mAP))))
            print('\tAverage-mAP: {}'.format(self.average_mAP))
        return self.mAP


def compute_average_precision_detection2(ground_truth, prediction):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'frame', 'score']
    Outputs
    -------
    ap : float
        Average precision score.
    """
    if prediction.empty:
        return 0

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros(len(prediction))
    fp = np.zeros(len(prediction))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(
                this_pred['video-id'])
        except Exception as e:
            fp[idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        flags = check_frame(this_pred[['frame']].values,
                            this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        match_sorted_idx = flags.argsort()[::-1]
        for jdx in match_sorted_idx:
            if flags[jdx] < 1:
                fp[idx] = 1
                break
            if lock_gt[this_gt.loc[jdx]['index']] >= 0:
                continue
            # Assign as true positive after the filters above.
            tp[idx] = 1
            lock_gt[this_gt.loc[jdx]['index']] = idx
            break

        if fp[idx] == 0 and tp[idx] == 0:
            fp[idx] = 1

    tp_cumsum = np.cumsum(tp).astype(np.float)
    fp_cumsum = np.cumsum(fp).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = interpolated_prec_rec(precision_cumsum, recall_cumsum)

    return ap


def compute_average_precision_detection(ground_truth, prediction, drange=np.linspace(0.25, 1, 4)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'frame', 'score']
    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(drange))
    if prediction.empty:
        return 0

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(drange), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(drange), len(prediction)))
    fp = np.zeros((len(drange), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(
                this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        distance = distance_2_center(this_pred[['frame']].values,
                                     this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        match_sorted_idx = distance.argsort()
        for tidx, td_thr in enumerate(drange):
            for jdx in match_sorted_idx:
                if distance[jdx] > td_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(drange)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :])

    return ap
