"""八码本的延迟排列；start=1 与 Dataset / Omni.generate 的首路启动位置一致。"""

import torch
from torch import Tensor


def delay_codes(
    codes: Tensor, pad_token: int = 2049, stop_token: int = 2050, start: int = 1
) -> Tensor:
    """Long [...,codebooks,frames] → [...,codebooks,frames+codebooks+start]，每路追加 stop。"""
    if codes.ndim < 2 or start < 0:
        raise ValueError("codes 至少包含 [codebooks,frames]；start 必须非负。")

    # 每路延迟一个位置，stop 紧跟该路最后一帧
    count, frames = codes.shape[-2:]
    delayed = codes.new_full((*codes.shape[:-1], frames + count + start), pad_token)
    for index in range(count):
        begin = start + index
        delayed[..., index, begin : begin + frames] = codes[..., index, :]
        delayed[..., index, begin + frames] = stop_token
    return delayed


def restore_codes(
    delayed: Tensor, frames: int, start: int = 1, codebook_size: int = 2048
) -> Tensor:
    """按已知有效帧数去掉 delay/stop，精确恢复 [...,codebooks,frames]。"""
    count = delayed.shape[-2]
    if frames < 0 or start < 0 or start + count - 1 + frames > delayed.shape[-1]:
        raise ValueError("有效帧范围超出延迟时间轴。")

    codes = torch.stack(
        [delayed[..., index, start + index : start + index + frames] for index in range(count)],
        dim=-2,
    )
    if ((codes < 0) | (codes >= codebook_size)).any():
        raise ValueError("还原范围含 stop/pad，frames 或 start 与原始排列不一致。")
    return codes


def code_mask(delayed: Tensor, codebook_size: int = 2048) -> Tensor:
    """Bool [...,codebooks,time]，只标记真实声学码本；stop/pad 均为 False。"""
    return (delayed >= 0) & (delayed < codebook_size)
