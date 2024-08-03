import os
import numpy as np
import matplotlib.pyplot as plt

class Logger():
    def __init__(self,name =None, path = None) -> None:
        self._value_curve = []
        self._iter = []
        if name is not None:
            name = name+' '
        else:
            name = ''
        self.name =name
        self.path = path

    def add_value(self,loss):
        self._value_curve.append(loss)
        self._iter.append(len(self._value_curve))


    def plot(self):
        plt.clf()

        # plot curve
        if len(self._value_curve) > 0:
            iter = np.array(self._iter)
            plt.plot(iter, np.array(self._value_curve), '-r')
        plt.title(self.name+' curve')
        plt.xlabel('iteration')
        plt.ylabel(self.name)
        if self.path is not None:
            plt.savefig(os.path.join(self.path,self.name))
        else:
            plt.savefig(self.name)