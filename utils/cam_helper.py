import pdb

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False,
                 ignore_index=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= high_thre] = ignore_index
        _pseudo_label[cam_value <= low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return valid_cam, pseudo_label


def cam_to_label_dynamic_cls(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False,
                 ignore_index=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= high_thre.unsqueeze(-1).unsqueeze(-1)] = ignore_index
        _pseudo_label[cam_value <= low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return valid_cam, pseudo_label


def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)
    # _pseudo_label += 1
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value <= low_thre] = 0
    roi_mask[cam_value >= hig_thre] = 2

    return roi_mask


def get_valid_cam(cam, cls_label):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam

    return valid_cam


def ignore_img_box(label, img_box, ignore_index):
    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label


def crop_from_roi_neg(images, roi_mask=None, crop_num=8, crop_size=96):
    crops = []

    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    flags = torch.ones(size=(b, crop_num + 2)).to(images.device)
    margin = crop_size // 2

    for i1 in range(b):
        roi_index = (roi_mask[i1, margin:(h - margin), margin:(w - margin)] <= 1).nonzero()
        if roi_index.shape[0] < crop_num:
            roi_index = (roi_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            temp_mask = roi_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            if temp_mask.sum() / (crop_size * crop_size) <= 0.2:
                ## if ratio of uncertain regions < 0.2 then negative
                flags[i1, i2 + 2] = 0

    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1, )
    crops = [c[:, 0] for c in _crops]

    return crops, flags


def multi_scale_cam2(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux


def multi_scale_cam2_siamese0(model, inputs, scales, branch=1):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    inputs_A, inputs_B = inputs
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True, branch=branch)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs_A = F.interpolate(inputs_A, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
                _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)

                _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True, branch=branch)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    return cam


def multi_scale_cam2_siamese1(model, inputs, scales, branch=1):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    inputs_A, inputs_B = inputs
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True, branch=branch)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs_A = F.interpolate(inputs_A, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
                _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)

                _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True, branch=branch)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    return cam


def multi_scale_cam2_siamese3(model, inputs, scales, branch=1):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    inputs_A, inputs_B = inputs
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True, branch=branch)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs_A = F.interpolate(inputs_A, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
                _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_A_cat, inputs_B_cat, cam_only=True, branch=branch)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux


def multi_scale_cam2_siamese_diff(model, inputs, scales, branch=1):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        if branch == 1:
            _cam_aux, _cam = model(x1=inputs_cat, cam_only=True, branch=branch)
        else:
            _cam_aux, _cam = model(x2=inputs_cat, cam_only=True, branch=branch)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                if branch == 1:
                    _cam_aux, _cam = model(x1=inputs_cat, cam_only=True, branch=branch)
                else:
                    _cam_aux, _cam = model(x2=inputs_cat, cam_only=True, branch=branch)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux


def multi_scale_cam2_siamese_both(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam_aux_1, _cam_1, _cam_aux_2, _cam_2 = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam_1, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux_1, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list_1 = [F.relu(_cam)]
        cam_aux_list_1 = [F.relu(_cam_aux)]

        _cam = F.interpolate(_cam_2, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux_2, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list_2 = [F.relu(_cam)]
        cam_aux_list_2 = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux_1, _cam_1, _cam_aux_2, _cam_2 = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam_1, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux_1, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list_1.append(F.relu(_cam))
                cam_aux_list_1.append(F.relu(_cam_aux))

                _cam = F.interpolate(_cam_2, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux_2, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list_2.append(F.relu(_cam))
                cam_aux_list_2.append(F.relu(_cam_aux))

        cam_1 = torch.sum(torch.stack(cam_list_1, dim=0), dim=0)
        cam_1 = cam_1 + F.adaptive_max_pool2d(-cam_1, (1, 1))
        cam_1 /= F.adaptive_max_pool2d(cam_1, (1, 1)) + 1e-5

        cam_aux_1 = torch.sum(torch.stack(cam_aux_list_1, dim=0), dim=0)
        cam_aux_1 = cam_aux_1 + F.adaptive_max_pool2d(-cam_aux_1, (1, 1))
        cam_aux_1 /= F.adaptive_max_pool2d(cam_aux_1, (1, 1)) + 1e-5

        cam_2 = torch.sum(torch.stack(cam_list_2, dim=0), dim=0)
        cam_2 = cam_2 + F.adaptive_max_pool2d(-cam_2, (1, 1))
        cam_2 /= F.adaptive_max_pool2d(cam_2, (1, 1)) + 1e-5

        cam_aux_2 = torch.sum(torch.stack(cam_aux_list_2, dim=0), dim=0)
        cam_aux_2 = cam_aux_2 + F.adaptive_max_pool2d(-cam_aux_2, (1, 1))
        cam_aux_2 /= F.adaptive_max_pool2d(cam_aux_2, (1, 1)) + 1e-5

    return cam_1, cam_aux_1, cam_2, cam_aux_2


def label_to_aff_mask(cam_label, ignore_index=255):
    b, h, w = cam_label.shape

    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)

    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index
    aff_label[:, range(h * w), range(h * w)] = ignore_index
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None,
                            ignore_index=False, img_box=None, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def refine_cams_with_dynamic_thres(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre_map=None,
                                   low_thre=None, ignore_index=False, img_box=None, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = high_thre_map
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


def cams_to_affinity_label(cam_label, mask=None, ignore_index=255):
    b, h, w = cam_label.shape

    cam_label_resized = F.interpolate(
        cam_label.unsqueeze(1).type(torch.float32), size=[h // 16, w // 16], mode="nearest"
    )

    _cam_label = cam_label_resized.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1).contiguous()
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    # aff_label[(_cam_label_rep+_cam_label_rep_t) == 0] = ignore_index
    for i in range(b):

        if mask is not None:
            aff_label[i, mask == 0] = ignore_index

        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index

    return aff_label


def crop_from_pseudo_label_1(images, pseudo_mask=None, crop_num=2, crop_size=128):
    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    margin = crop_size // 2

    # Exp
    crop_pseudo_label = torch.zeros(size=(b, crop_num, 1, crop_size, crop_size)).to(images.device)

    for i1 in range(b):
        roi_index = (
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 0) &
                (pseudo_mask[i1, margin:(h - margin), margin:(w - margin)] != 255)
        ).nonzero()

        if roi_index.shape[0] < crop_num:
            roi_index = (pseudo_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            crop_pseudo_label[i1, i2, ...] = pseudo_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)]

    return temp_crops, crop_pseudo_label


def pseudo_label_selection(pseudo_label_1, pseudo_label_2):
    same_pred = (pseudo_label_1 == pseudo_label_2) * (pseudo_label_1 != 255) * (pseudo_label_2 != 255)
    return same_pred


# select common uncertain regions
def pseudo_uncertain_selection(pseudo_label_1, pseudo_label_2, img_box=None, ignore_index=255):
    b, h, w = pseudo_label_1.shape
    same_pred = (pseudo_label_1 == ignore_index) * (pseudo_label_2 == ignore_index)

    if img_box is not None:
        same_pred_box = torch.zeros(size=(b, h, w))

        for idx, coord in enumerate(img_box):
            same_pred_box[idx, coord[0]:coord[1], coord[2]:coord[3]] = same_pred[0, coord[0]:coord[1],
                                                                       coord[2]:coord[3]]
        return same_pred_box

    return same_pred


# select common regions
def pseudo_same_selection(pseudo_label_1, pseudo_label_2, img_box=None, ignore_index=255):
    b, h, w = pseudo_label_1.shape
    same_pred = (pseudo_label_1 == pseudo_label_2) * (pseudo_label_1 != ignore_index) * (pseudo_label_2 != ignore_index)

    if img_box is not None:
        same_pred_box = torch.zeros(size=(b, h, w))

        for idx, coord in enumerate(img_box):
            same_pred_box[idx, coord[0]:coord[1], coord[2]:coord[3]] = same_pred[0, coord[0]:coord[1],
                                                                       coord[2]:coord[3]]
        return same_pred_box

    return same_pred
