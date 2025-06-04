import torch
import numpy as np


def reliable_pseudo_label(seg1_label, seg2_label, cam1_label, cam2_label, cls_label):
    batch_size = cam1_label.shape[0]
    mIOU1 = []
    for i in range(batch_size):
        metric = meanIOU(num_classes=2)
        metric.add_batch(seg1_label[i].cpu().numpy(), cam2_label[i].cpu().numpy())
        mIOU1.append(metric.evaluate()[-1])
    sorted_indexes1 = sorted(range(len(mIOU1)), key=lambda i : mIOU1[i])
    cam1_label[sorted_indexes1[: batch_size // 2], :, :] = 255
    reliable_index1 = sorted_indexes1[batch_size // 2:]

    mIOU2 = []
    for i in range(batch_size):
        metric = meanIOU(num_classes=2)
        metric.add_batch(seg2_label[i].cpu().numpy(), cam1_label[i].cpu().numpy())
        mIOU2.append(metric.evaluate()[-1])
    sorted_indexes2 = sorted(range(len(mIOU2)), key=lambda i: mIOU2[i])
    cam2_label[sorted_indexes2[: batch_size // 2], :, :] = 255
    reliable_index2 = sorted_indexes2[batch_size // 2:]
    return cam1_label, cam2_label, reliable_index1, reliable_index2


def reliable_pseudo_label2(seg1_label, seg2_label, cam1_label, cam2_label, cls_label):
    batch_size, height, width = seg1_label.shape
    seg1_label = seg1_label.reshape(batch_size, 4, height // 4, 4, width // 4).permute(0, 1, 3, 2, 4)
    seg1_label = seg1_label.reshape(-1, height // 4, width // 4)
    seg2_label = seg2_label.reshape(batch_size, 4, height // 4, 4, width // 4).permute(0, 1, 3, 2, 4)
    seg2_label = seg2_label.reshape(-1, height // 4, width // 4)
    cam1_label = cam1_label.reshape(batch_size, 4, height // 4, 4, width // 4).permute(0, 1, 3, 2, 4)
    cam1_label = cam1_label.reshape(-1, height // 4, width // 4)
    cam2_label = cam2_label.reshape(batch_size, 4, height // 4, 4, width // 4).permute(0, 1, 3, 2, 4)
    cam2_label = cam2_label.reshape(-1, height // 4, width // 4)

    mIOU1 = []
    for i in range(batch_size * 4 * 4):
        metric = meanIOU(num_classes=2)
        metric.add_batch(seg1_label[i], cam2_label[i])
        mIOU1.append(metric.evaluate()[-1])
    sorted_indexes1 = sorted(range(len(mIOU1)), key=lambda i : mIOU1[i])
    seg1_label[sorted_indexes1[: batch_size // 2], :, :] = 255
    unreliable_index1 = sorted_indexes1[: batch_size // 2]

    mIOU2 = []
    for i in range(batch_size):
        metric = meanIOU(num_classes=2)
        metric.add_batch(seg2_label[i], cam1_label[i])
        mIOU2.append(metric.evaluate()[-1])
    sorted_indexes2 = sorted(range(len(mIOU2)), key=lambda i: mIOU1[i])
    seg2_label[sorted_indexes2[: batch_size // 2], :, :] = 255
    unreliable_index2 = sorted_indexes2[: batch_size // 2]
    return seg1_label, seg2_label, unreliable_index1, unreliable_index2


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def batch_pix_accuracy(predict, target):

    # _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()