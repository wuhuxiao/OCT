import numpy as np

start = 30
end = 1520
intervals = [0, 30.5, 31.5, 32.5, 33.5, 34.5, 35.5, 36.5, 37.5, 38.5, 39.5]


def getInterval(start):
    intervals.append(start)
    if start > end:
        return
    getInterval(start * (1 + 0.025))


getInterval(40)
intervals = np.array(intervals)


def getOneHot(num):
    index = (intervals > num).argmax() - 1
    NUM_CLASSES = len(intervals) - 1
    levels = [1] * index + [0] * (NUM_CLASSES - 1 - index)
    return index,levels


def getValue(index):
    if index :
        return (intervals[index] + intervals[index + 1]) / 2
    else:
        return 30

