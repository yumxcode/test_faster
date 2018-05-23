import numpy as np
a=np.array([[1,2,3],
            [2,3,4],
            [3,4,5],
            [5,6,7],
            [7,8,9]])
argmax=a.argmax(axis=1)
max_overlaps=a[np.arange(5),argmax]
label=np.empty((5,),dtype=np.float32)
label.fill(-1)
label[max_overlaps<5]=0   #notice that label the index 0 if this index's value of max_overlaps is less than 5
print(argmax)
print(max_overlaps)
print(label)
