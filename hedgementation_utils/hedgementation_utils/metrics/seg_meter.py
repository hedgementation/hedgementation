import torch

class SegMeter(object):
    """
    Score semantic segmentation metrics by accumulating histogram
    of outputs and targets.
    - overall pixel accuracy
    - per-class accuracy
    - per-class intersection-over-union (iou)
    - frequency weighted iou
    n.b. mean iou is the standard single number summary of segmentation quality.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = None

    def update(self, output, target):
        if self.hist is None:
            # init on first update (to inherit target device)
            self.hist = target.new_zeros((self.num_classes,) * 2).float()
        # metrics require hard assigment, so pick most likely class
        self.hist += self.fast_hist(target, output, self.num_classes)

    def update_batch(self,output,target,batch_dim=0):
        """
        Iterates over all of the (output, target) pairs along the batch dim, 
        and calls self.update() on each
        """
        batch_size = output.size(batch_dim)
        for i in range(batch_size):
            output_i = output.select(batch_dim, i)
            target_i = target.select(batch_dim, i)
            self.update(output_i, target_i)

    def score(self):
        scores = {}
        # overall accuracy
        scores['all_acc'] = torch.diag(self.hist).sum() / self.hist.sum()
        # per-class accuracy
        scores['acc'] = torch.diag(self.hist) / self.hist.sum(1)

        # per-class precision
        scores['pre'] = torch.diag(self.hist) / self.hist.sum(0) #sum over row to sum true and false positives
        scores['f1'] = 2 * (scores['acc'] * scores['pre']) / (scores['acc'] + scores['pre']) #formula for f1
        #metrics are micro-averaged so all_pre and all_f1 are identical to all_acc

        # per-class iou
        scores['iou'] = torch.diag(self.hist) / (self.hist.sum(1)
                                             + self.hist.sum(0)
                                             - torch.diag(self.hist))
        # frequency-weighted iou
        freq = self.hist.sum(1) / self.hist.sum()
        scores['freq_iou'] = (freq[freq > 0] * scores['iou'][freq > 0]).sum()
        return scores
    
    def ask_metrics(self,cl=1):
        #quick helper method to return performance for a specific class, defaults to 1
        
        scores = self.score()
        retval = {}

        for m in scores:
            if scores[m].dim() != 0:
                retval[m] = scores[m][cl].item()
        return retval


    def fast_hist(self, a, b, n):
        """
        Fast 2D histogram by linearizing.
        """
        a, b = a.view(-1).long(), b.view(-1).long()
        k = (a >= 0) & (a < n)
        return torch.bincount(n * a[k] + b[k],
                              minlength=n**2).view(n, n).float()

    def __str__(self):
        strs = []
        for metric, score in self.score().items():
            score = torch.mean(score[score == score]).item()  # filter out NaN
            strs.append(f"{metric:10s} {score:.3f}")
        return '\n'.join(strs)