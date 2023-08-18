# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample, TransformerBlock, PatchEmbeddingBlock, UnetrBasicBlock, ADN, ResidualUnit, Convolution
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from monai.networks.nets.swin_unetr import SwinTransformer, PatchMerging, PatchMergingV2
from typing import Optional, Sequence, Tuple, Type, Union
import numpy as np
from monai.utils import ensure_tuple_rep, look_up_option, optional_import



class Down_res(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        max_pooling = Pool["MAX", dim](kernel_size=2)
        #convs = TwoConv(dim, in_chns, out_chns, act, norm, bias, dropout)

        convs = ResidualUnit(dim, in_chns, out_chns,
                             strides=1, kernel_size=3, subunits=2,
                             adn_ordering='NDA', act=act, norm=norm,
                             bias=bias, dropout=dropout
                             )
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat_res(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
        """
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            dim,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = ResidualUnit(dim, cat_chns + up_chns, out_chns,
                             strides=1, kernel_size=3, subunits=2,
                             adn_ordering='NDA', act=act, norm=norm,
                             bias=bias, dropout=dropout
                             )

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x



MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}
class cats_v2(nn.Module):
    def __init__(
        self,
            img_size: Union[Sequence[int], int],
            in_channels: int,
            out_channels: int,
            depths: Sequence[int] = (2, 2, 2, 2),
            num_heads: Sequence[int] = (3, 6, 12, 24),
            feature_size: int = 24,
            norm_name: Union[Tuple, str] = "instance",
            drop_rate: float = 0.0,
            attn_drop_rate: float = 0.0,
            dropout_path_rate: float = 0.0,
            normalize: bool = True,
            use_checkpoint: bool = False,
            spatial_dims: int = 3,
            downsample="merging",

            # CNN part below
            features: Sequence[int] = (32, 32, 64, 64, 128, 256, 32),
            act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: Union[str, tuple] = ("instance", {"affine": True}),
            bias: bool = True,
            dropout: Union[float, tuple] = 0.0,
            upsample: str = "deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.

        Based on:

            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2

        Args:
            dimensions: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: six integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,

                - the first five values correspond to the five-level encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.

            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.

        Examples::

            # for spatial 2D
            >>> net = cats_v2(dimensions=2, features=(64, 64, 128, 256, 512, 1024, 128))

            # for spatial 2D, with group norm
            >>> net = cats_v2(dimensions=2, features=(64, 64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))

            # for spatial 3D
            >>> net = cats_v2(dimensions=3, features=(32, 32, 32, 64, 128, 256, 32))

        See Also

            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`

        """
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
        self.normalize = normalize


        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        )



        fea = ensure_tuple_rep(features, 7)
        print(f"cats_v2 features: {fea}.")

        self.conv_0 = ResidualUnit(spatial_dims, in_channels, features[0],
                                   strides=1, kernel_size=3, subunits=2,
                                   adn_ordering='NDA', act=act, norm=norm,
                                   bias=bias, dropout=dropout
                                   )

        self.down_1 = Down_res(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down_res(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down_res(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down_res(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        self.down_5 = Down_res(spatial_dims, fea[4], fea[5], act, norm, bias, dropout)

        self.upcat_5 = UpCat_res(spatial_dims, fea[5], fea[4], fea[4], act, norm, bias, dropout, upsample)
        self.upcat_4 = UpCat_res(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat_res(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat_res(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat_res(spatial_dims, fea[1], fea[0], fea[6], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", spatial_dims](fea[6], out_channels, kernel_size=1)

        self.bottom_conv = ResidualUnit(spatial_dims, fea[5], fea[5],
                                        strides=1, kernel_size=3, subunits=2,
                                        adn_ordering='NDA', act='RELU', norm='batch')



        self.fourth_conv = ResidualUnit(spatial_dims, fea[4], fea[4],
                                        strides=1, kernel_size=3, subunits=2,
                                        adn_ordering='NDA', act='RELU', norm='batch')

        self.third_conv = ResidualUnit(spatial_dims, fea[3], fea[3],
                                        strides=1, kernel_size=3, subunits=2,
                                        adn_ordering='NDA', act='RELU', norm='batch')

        self.second_conv = ResidualUnit(spatial_dims, fea[2], fea[2],
                                       strides=1, kernel_size=3, subunits=2,
                                       adn_ordering='NDA', act='RELU', norm='batch')

        self.first_conv = ResidualUnit(spatial_dims, fea[1], fea[1],
                                        strides=1, kernel_size=3, subunits=2,
                                        adn_ordering='NDA', act='RELU', norm='batch')


    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        x5 = self.down_5(x4)

        hidden_states_out = self.swinViT(x, self.normalize)

        # swin
        swin_4 = self.bottom_conv(hidden_states_out[4])
        swin_3 = self.fourth_conv(hidden_states_out[3])
        swin_2 = self.third_conv(hidden_states_out[2])
        swin_1 = self.second_conv(hidden_states_out[1])
        swin_0 = self.first_conv(hidden_states_out[0])

        x5 = x5 + swin_4
        x4 = x4 + swin_3
        x3 = x3 + swin_2
        x2 = x2 + swin_1
        x1 = x1 + swin_0

        u5 = self.upcat_5(x5, x4)
        u4 = self.upcat_4(u5, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        return logits


