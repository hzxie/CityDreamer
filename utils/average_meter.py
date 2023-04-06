# -*- coding: utf-8 -*-
#
# @File:   average_meter.py
# @Author: Haozhe Xie
# @Date:   2019-08-06 22:50:12
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-04-06 10:07:14
# @Email:  root@haozhexie.com


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, items=None):
        self.items = items
        self.n_items = 1 if items is None else len(items)
        self.reset()

    def reset(self):
        self._val = [0] * self.n_items
        self._sum = [0] * self.n_items
        self._count = [0] * self.n_items

    def update(self, values, weight=1):
        if type(values).__name__ == "list":
            for idx, v in enumerate(values):
                self._val[idx] = v
                self._sum[idx] += v * weight
                self._count[idx] += weight
        else:
            self._val[0] = values
            self._sum[0] += values * weight
            self._count[0] += weight

    def val(self, idx=None):
        if idx is None:
            return (
                self._val[0]
                if self.items is None
                else [self._val[i] for i in range(self.n_items)]
            )
        else:
            return self._val[idx]

    def count(self, idx=None):
        if idx is None:
            return (
                self._count[0]
                if self.items is None
                else [self._count[i] for i in range(self.n_items)]
            )
        else:
            return self._count[idx]

    def avg(self, idx=None):
        if idx is None:
            return (
                self._sum[0] / self._count[0]
                if self.items is None
                else [self._sum[i] / self._count[i] for i in range(self.n_items)]
            )
        else:
            return self._sum[idx] / self._count[idx]
