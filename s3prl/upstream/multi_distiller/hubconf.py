"""
    hubconf for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os
from .expert import UpstreamExpert as _UpstreamExpert


def multi_distiller_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    # ckpt = '/work/u8786328/john/result/pretrain/humert_lconv_1/states-100000.ckpt'
    #ckpt = "/home/johnwei743251/s3prl/s3prl/result/pretrain/wavlm+mert_origin/states-epoch-14.ckpt"
    print('=========================================',ckpt)
    #assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def distilhubert(refresh=False, *args, **kwargs):
    """
    DistilHuBERT
    """
    return distilhubert_base(refresh=refresh, *args, **kwargs)
