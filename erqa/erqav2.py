import numpy as np
import cv2

class ERQA:
    def __init__(self, cosine_similarity_thr=0.85, grad_percentile=0.85, num_shifts=35, beta=0.1, window_range=5):
        """
            ERQAv2 - Edge Restoration Quality Assessment metric
            
            Args:
                cosine_similarity_thr (float): The threshold for the scalar product between the vectors of gt and sr image 
                                               above which the gradients are considered the same 
                grad_perc (float): The threshold of the norm of gradients that participate in the comparison
                num_shifts (int): The number of global shifts, which are considered during gradients comparison
                window_range (int): [-window_range, window_range] - the values of global shifts
                beta (float): positive real factor of f-score, where beta is chosen such that recall is considered beat times 
                              as important as precision
        """
        self.cosine_similarity_thr = cosine_similarity_thr
        self.grad_perc = grad_percentile
        self.num_shifts = num_shifts
        self.beta = beta
        self.window_range = window_range
        self.eps = 1e-8
    
    
    def __call__(self, image_sr, image_gt):
        assert image_gt.shape == image_sr.shape
        assert image_gt.shape[2] == 3, 'Compared images should be in RGB format'
        
        image_sr = cv2.cvtColor(image_sr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        image_gt = cv2.cvtColor(image_gt, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.
        
        grad_gt = self._get_grad(image_gt)
        grad_sr = self._get_grad(image_sr)
        
        return self._metric(grad_sr, grad_gt)
    

    def _get_grad(self, image):
        [grad_x, grad_y] = np.gradient(image) 
        grad_x = np.where(np.abs(grad_x) >= np.quantile(np.abs(grad_x), self.grad_perc), np.abs(grad_x), 0)
        grad_y = np.where(np.abs(grad_y) >= np.quantile(np.abs(grad_y), self.grad_perc), np.abs(grad_y), 0)
        return grad_x, grad_y
    
    
    def _norm(self, grad):
        return np.sqrt(np.square(grad[0]) + np.square(grad[1]))
    
    
    def _normalize(self, grad):
        return grad / (np.sqrt(np.square(grad[0]) + np.square(grad[1])) + self.eps)
    
    
    def _metric(self, grad_sr, grad_gt):
        grad_sr = self._normalize(grad_sr)
        grad_gt = self._normalize(grad_gt)
        
        window = sorted([(i, j) for i in range(-self.window_range, self.window_range + 1)
                                for j in range(-self.window_range, self.window_range + 1)], 
                        key=lambda x: abs(x[0]) + abs(x[1]))
        ads = []
        for (i, j) in window:
            grad_sr_shifted = np.roll(grad_sr, shift = (i, j), axis = (-2, -1))
            dot = (grad_gt[0] * grad_sr_shifted[0] + grad_gt[1] * grad_sr_shifted[1])
            ad = np.where(dot > self.cosine_similarity_thr, 1, 0)
            ads.append(np.sum(ad))

        window = np.array(window)[np.argsort(ads)][::-1]
        
        true_positive = np.zeros((grad_sr.shape[1], grad_sr.shape[2]))
        for [i, j] in window[:self.num_shifts]:
            grad_sr_shifted = np.roll(grad_sr, shift = (i, j), axis = (-2, -1))
            dot = (grad_gt[0] * grad_sr_shifted[0] + grad_gt[1] * grad_sr_shifted[1])
            ad = np.where(dot > self.cosine_similarity_thr, 1, 0)

            np.logical_or(true_positive, ad, out=true_positive)
            grad_gt *= np.logical_not(ad)
            ad = np.roll(ad, shift = (-i, -j), axis = (-2, -1))
            grad_sr *= np.logical_not(ad)

        false_negative = np.where(self._norm(grad_gt) > 0, 1, 0)
        false_positive = np.where(self._norm(grad_sr) > 0, 1, 0)

        return self.f_measure(true_positive, false_negative, false_positive)
    
    def f_measure(self, true_positive, false_negative, false_positive):
        tp = np.sum(true_positive)
        fp = np.sum(false_positive)
        fn = np.sum(false_negative)
    
        if tp == 0 or tp + fp == 0 or tp + fn == 0:
            f1 = 0
        else:
            beta = self.beta
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            f1 = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        return f1