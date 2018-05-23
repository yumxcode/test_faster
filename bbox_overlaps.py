import numpy as np
def bbox_overlaps(boxes,gt_boxes):
    N=boxes.shape[0]
    K=gt_boxes.shape[0]
    overlaps=np.zeros((N,K),dtype=np.float)
    for k in range(K):
        box_area=((gt_boxes[k,2]-gt_boxes[k,0]+1)*(gt_boxes[k,3]-gt_boxes[k,1]+1))
        for n in range(N):
            iw=(min(boxes[n,2],gt_boxes[k,2])-max(boxes[n,0],gt_boxes[k,0])+1)
            if iw>0:
                ih=(min(boxes[n,3],gt_boxes[k,3])-max(boxes[n,1],gt_boxes[k,1])+1)
                if ih>0:
                    ua=float((boxes[n,2]-boxes[n,0]+1)*
                             (boxes[n,3]-boxes[n,1]+1)+box_area-iw*ih)
                    overlaps[n,k]=iw*ih/ua
    return overlaps
