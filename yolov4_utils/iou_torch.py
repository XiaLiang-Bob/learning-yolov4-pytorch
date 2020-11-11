import torch
import numpy as np

def iou_torch(bboxes_a, bboxes_b, bbox_type="x1y1x2y2", iou_type='IoU'):
    if bbox_type is "x1y1x2y2":
        i_tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])#交集左上角坐标
        i_br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])#交集右下角坐标
        c_tl = torch.min(bboxes_a[:, None, :2], bboxes_b[:, :2])
        c_br = torch.max(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        dist_c = torch.pow(c_br - c_tl, 2).sum(dim=2) + 1e-16
        dist_b = torch.pow((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:])/2. - (bboxes_b[:, :2] + bboxes_b[:, 2:])/2., 2).sum(dim=2)
        area_a = (bboxes_a[:, 2:] - bboxes_a[:, :2]).prod(dim=1)
        area_b = (bboxes_b[:, 2:] - bboxes_b[:, :2]).prod(dim=1)
    elif bbox_type is "cxcywh":
        i_tl = torch.max(bboxes_a[:, None, :2] - bboxes_a[:, None, 2:]/2., bboxes_b[:, :2] - bboxes_b[:, 2:]/2.)
        i_br = torch.min(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:]/2., bboxes_b[:, :2] + bboxes_b[:, 2:]/2.)
        c_tl = torch.min(bboxes_a[:, None, :2] - bboxes_a[:, None, 2:]/2., bboxes_b[:, :2] - bboxes_b[:, 2:]/2.)
        c_br = torch.max(bboxes_a[:, None, :2] + bboxes_a[:, None, 2:]/2., bboxes_b[:, :2] + bboxes_b[:, 2:]/2.)
        dist_c = torch.pow(c_br - c_tl, 2).sum(dim=2) + 1e-16
        dist_b = torch.pow(bboxes_a[:, None, :2] - bboxes_b[:, None, :2], 2).sum(dim=2)
        area_a = torch.prod(bboxes_a[:, 2:], dim=1)
        area_b = torch.prod(bboxes_b[:, 2:], dim=1)
    else:
        print("bbox type set error: chose x1y1x2y2 or cxcywh")
        raise TypeError
    sign = (i_br > i_tl).prod(dim=2)
    area_i = (i_br-i_tl).prod(dim=2)*sign
    area_u = area_a[:, None] + area_b - area_i

    iou = area_i / area_u
    if iou_type is "IoU":
        return iou


    elif iou_type is "GIoU":
        area_c = torch.prod(c_br - c_tl, dim=2)
        giou = iou - (area_c - area_u) / area_c
        return giou

    elif iou_type is "DIoU":
        diou = iou - dist_b / dist_c
        return diou

    elif iou_type is "CIoU":
        w_a = bboxes_a[:, 2] - bboxes_a[:, 0]
        h_a = bboxes_a[:, 3] - bboxes_a[:, 1]
        w_b = bboxes_b[:, 2] - bboxes_b[:, 0]
        h_b = bboxes_b[:, 3] - bboxes_b[:, 1]
        v = (4. / np.pi**2) * torch.pow(torch.atan(w_a / h_a).unsqueeze(dim=1) - torch.atan(w_b / h_b), 2) + 1e-16
        alpha = v / (1 - iou + v)

        ciou = iou - dist_b / dist_c - alpha*v
        return ciou
    else:
        print("iou type error: chose from [IoU, GIoU, DIoU, CIoU]")
        raise TypeError

if __name__ == "__main__":
    bboxes_a = torch.FloatTensor([[40, 40, 80, 80],
                                 [20, 20, 40, 40]])
    bboxes_b = torch.FloatTensor([[40, 40, 80, 80],
                                 [50, 50, 90, 90],
                                 [20, 20, 40, 40],
                                 [50, 50, 80, 80]])

    iou = iou_torch(bboxes_a, bboxes_b, bbox_type="x1y1x2y2", iou_type='CIoU')
    iou_loss = 1-iou
    print(iou_loss)
