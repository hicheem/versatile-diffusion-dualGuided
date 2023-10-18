from abc import abstractmethod
from functools import partial
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np
import copy

from attention import SpatialTransformer
from diffusion_utils import \
    checkpoint, conv_nd, linear, avg_pool_nd, \
    zero_module, normalization, timestep_embedding

# Importing predefined blocks from the 'openaimodel' module within the Hugging Face library.

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class Linear_MultiDim(nn.Linear):
    def __init__(self, in_features, out_features, *args, **kwargs):
        
        in_features = [in_features] if isinstance(in_features, int) else list(in_features)
        out_features = [out_features] if isinstance(out_features, int) else list(out_features)
        self.in_features_multidim = in_features
        self.out_features_multidim = out_features
        super().__init__(
            np.array(in_features).prod(), 
            np.array(out_features).prod(), 
            *args, **kwargs)

    def forward(self, x):
        shape = x.shape
        n = len(shape) - len(self.in_features_multidim)
        x = x.view(*shape[:n], self.in_features)
        y = super().forward(x)
        y = y.view(*shape[:n], *self.out_features_multidim)
        return y

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class FCBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 1, padding=0),)

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, self.out_channels,),)
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 1, padding=0)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1, padding=0)

    def forward(self, x, emb):
        if len(x.shape) == 2:
            x = x[:, :, None, None]
        elif len(x.shape) == 4:
            pass
        else:
            raise ValueError
        y = checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        if len(x.shape) == 2:
            return y[:, :, 0, 0]
        elif len(x.shape) == 4:
            return y

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class FCBlock_MultiDim(FCBlock):
    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channels=None,
            use_checkpoint=False,):
        channels = [channels] if isinstance(channels, int) else list(channels)
        channels_all = np.array(channels).prod()
        self.channels_multidim = channels

        if out_channels is not None:
            out_channels = [out_channels] if isinstance(out_channels, int) else list(out_channels)
            out_channels_all = np.array(out_channels).prod()
            self.out_channels_multidim = out_channels
        else:
            out_channels_all = channels_all
            self.out_channels_multidim = self.channels_multidim

        self.channels = channels
        super().__init__(
            channels = channels_all,
            emb_channels = emb_channels,
            dropout = dropout,
            out_channels = out_channels_all,
            use_checkpoint = use_checkpoint,)

    def forward(self, x, emb):
        shape = x.shape
        n = len(self.channels_multidim)
        x = x.view(*shape[0:-n], self.channels, 1, 1)
        x = x.view(-1, self.channels, 1, 1)
        y = checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint)
        y = y.view(*shape[0:-n], -1)
        y = y.view(*shape[0:-n], *self.out_channels_multidim)
        return y


# U-Net model tailored for image processing, representing the first part of the Versatile Diffusion Model.
class UNetModel2D_Next(nn.Module):
    def __init__(
            self,
            in_channels=4,
            model_channels=320,
            out_channels=4,
            num_res_blocks=[ 2, 2, 2, 2 ],
            attention_resolutions=[ 4, 2, 1 ],
            context_dim=768,
            dropout=0,
            channel_mult=(1, 2, 4, 4),
            conv_resample=True,
            use_checkpoint=False,
            num_heads=8,
            num_head_channels=None,
            parts = ['global', 'data', 'context']):

        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks

        self.attention_resolutions = attention_resolutions
        self.context_dim = context_dim
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        assert (num_heads is None) + (num_head_channels is None) == 1, \
            "One of num_heads or num_head_channels need to be set"

        self.parts = parts if isinstance(parts, list) else [parts]
        self.glayer_included = 'global' in self.parts
        self.dlayer_included = 'data' in self.parts
        self.clayer_included = 'context' in self.parts
        self.layer_sequence_ordering = []

        #################
        # global layers #
        #################

        time_embed_dim = model_channels * 4
        if self.glayer_included:
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        ################
        # input layers #
        ################

        if self.dlayer_included:
            self.data_blocks = nn.ModuleList([])
            ResBlockDefault = partial(
                ResBlock, 
                emb_channels=time_embed_dim,
                dropout=dropout,
                dims=2,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=False, )
        else:
            def dummy(*args, **kwargs):
                return None
            ResBlockDefault = dummy

        if self.clayer_included:
            self.context_blocks = nn.ModuleList([])
            CrossAttnDefault = partial(
                SpatialTransformer, 
                context_dim=context_dim,
                disable_self_attn=False, )
        else:
            def dummy(*args, **kwargs):
                return None
            CrossAttnDefault = dummy

        self.add_data_layer(conv_nd(2, in_channels, model_channels, 3, padding=1))
        self.layer_sequence_ordering.append('save_hidden_feature')
        input_block_chans = [model_channels]

        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layer = ResBlockDefault(
                    channels=ch, out_channels=mult*model_channels,)
                self.add_data_layer(layer)
                ch = mult * model_channels

                if (ds in attention_resolutions):
                    d_head, n_heads = self.get_d_head_n_heads(ch)
                    layer = CrossAttnDefault(
                        in_channels=ch, d_head=d_head, n_heads=n_heads,)
                    self.add_context_layer(layer)
                input_block_chans.append(ch)
                self.layer_sequence_ordering.append('save_hidden_feature')

            if level != len(channel_mult) - 1:
                layer = Downsample(
                    ch, use_conv=True, dims=2, out_channels=ch)
                self.add_data_layer(layer)
                input_block_chans.append(ch)
                self.layer_sequence_ordering.append('save_hidden_feature')
                ds *= 2

        self.i_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        #################
        # middle layers #
        #################

        self.add_data_layer(ResBlockDefault(channels=ch))
        d_head, n_heads = self.get_d_head_n_heads(ch)
        self.add_context_layer(CrossAttnDefault(in_channels=ch, d_head=d_head, n_heads=n_heads))
        self.add_data_layer(ResBlockDefault(channels=ch))

        self.m_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        #################
        # output layers #
        #################

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for _ in range(self.num_res_blocks[level] + 1):
                self.layer_sequence_ordering.append('load_hidden_feature')
                ich = input_block_chans.pop()
                layer = ResBlockDefault(
                    channels=ch+ich, out_channels=model_channels*mult,)
                ch = model_channels * mult
                self.add_data_layer(layer)

                if ds in attention_resolutions:
                    d_head, n_heads = self.get_d_head_n_heads(ch)
                    layer = CrossAttnDefault(
                        in_channels=ch, d_head=d_head, n_heads=n_heads)
                    self.add_context_layer(layer)

            if level != 0:
                layer = Upsample(ch, conv_resample, dims=2, out_channels=ch)
                self.add_data_layer(layer)
                ds //= 2                

        layer = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(2, model_channels, out_channels, 3, padding=1)),
        )
        self.add_data_layer(layer)

        self.o_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_order = copy.deepcopy(self.i_order + self.m_order + self.o_order)
        del self.layer_sequence_ordering

        self.parameter_group = {}
        if self.glayer_included:
            self.parameter_group['global'] = self.time_embed
        if self.dlayer_included:
            self.parameter_group['data'] = self.data_blocks
        if self.clayer_included:
            self.parameter_group['context'] = self.context_blocks

    def get_d_head_n_heads(self, ch):
        if self.num_head_channels is None:
            d_head = ch // self.num_heads
            n_heads = self.num_heads
        else:
            d_head = self.num_head_channels
            n_heads = ch // self.num_head_channels
        return d_head, n_heads

    def add_data_layer(self, layer):
        if self.dlayer_included:
            if not isinstance(layer, (list, tuple)):
                layer = [layer]
            self.data_blocks.append(TimestepEmbedSequential(*layer))
        self.layer_sequence_ordering.append('d')

    def add_context_layer(self, layer):
        if self.clayer_included:
            if not isinstance(layer, (list, tuple)):
                layer = [layer]
            self.context_blocks.append(TimestepEmbedSequential(*layer))
        self.layer_sequence_ordering.append('c')

    def forward(self, x, timesteps, context):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        d_iter = iter(self.data_blocks)
        c_iter = iter(self.context_blocks)

        h = x
        for ltype in self.i_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, context)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, context)
            elif ltype == 'save_hidden_feature':
                hs.append(h)

        for ltype in self.m_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, context)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, context)

        for ltype in self.i_order:
            if ltype == 'load_hidden_feature':
                h = th.cat([h, hs.pop()], dim=1)
            elif ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, context)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, context)
        o = h

        return o

# U-Net model tailored for text processing, representing the second part of the Versatile Diffusion Model.
class UNetModel0D_Next(UNetModel2D_Next):
    def __init__(
            self,
            input_channels=768,
            model_channels=320,
            output_channels=768,
            context_dim = 768,
            num_noattn_blocks=(2, 2, 2, 2),
            channel_mult=(1, 2, 4, 4),
            second_dim=(4, 4, 4, 4),
            with_attn=[True, True, True, False],
            num_heads=8,
            num_head_channels=None,
            use_checkpoint=False,
            parts = ['global', 'data', 'context']):

        super(UNetModel2D_Next, self).__init__()

        self.input_channels = input_channels
        self.model_channels = model_channels
        self.output_channels = output_channels
        self.num_noattn_blocks = num_noattn_blocks
        self.channel_mult = channel_mult
        self.second_dim = second_dim
        self.with_attn = with_attn
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        self.parts = parts if isinstance(parts, list) else [parts]
        self.glayer_included = 'global' in self.parts
        self.dlayer_included = 'data' in self.parts
        self.clayer_included = 'context' in self.parts
        self.layer_sequence_ordering = []

        #################
        # global layers #
        #################

        time_embed_dim = model_channels * 4
        if self.glayer_included:
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        ################
        # input layers #
        ################

        if self.dlayer_included:
            self.data_blocks = nn.ModuleList([])
            FCBlockDefault = partial(
                FCBlock_MultiDim, dropout=0, use_checkpoint=use_checkpoint)
        else:
            def dummy(*args, **kwargs):
                return None
            FCBlockDefault = dummy

        if self.clayer_included:
            self.context_blocks = nn.ModuleList([])
            CrossAttnDefault = partial(
                SpatialTransformer, 
                context_dim=context_dim,
                disable_self_attn=False, )
        else:
            def dummy(*args, **kwargs):
                return None
            CrossAttnDefault = dummy

        sdim = second_dim[0]
        current_channel = [model_channels, sdim, 1]
        one_layer = Linear_MultiDim([input_channels], current_channel, bias=True)
        self.add_data_layer(one_layer)
        self.layer_sequence_ordering.append('save_hidden_feature')
        input_block_channels = [current_channel]

        for level_idx, (mult, sdim) in enumerate(zip(channel_mult, second_dim)):
            for _ in range(self.num_noattn_blocks[level_idx]):
                layer = FCBlockDefault(
                    current_channel, 
                    time_embed_dim,
                    out_channels = [mult*model_channels, sdim, 1],)

                self.add_data_layer(layer)
                current_channel = [mult*model_channels, sdim, 1]

                if with_attn[level_idx]:
                    d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
                    layer = CrossAttnDefault(
                        in_channels=current_channel[0],
                        d_head=d_head, n_heads=n_heads,)
                    self.add_context_layer(layer)

                input_block_channels.append(current_channel)
                self.layer_sequence_ordering.append('save_hidden_feature')

            if level_idx != len(channel_mult) - 1:
                layer = Linear_MultiDim(current_channel, current_channel, bias=True,)
                self.add_data_layer(layer)
                input_block_channels.append(current_channel)
                self.layer_sequence_ordering.append('save_hidden_feature')

        self.i_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        #################
        # middle layers #
        #################

        self.add_data_layer(FCBlockDefault(current_channel, time_embed_dim, ))
        d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
        self.add_context_layer(CrossAttnDefault(in_channels=current_channel[0], d_head=d_head, n_heads=n_heads))
        self.add_data_layer(FCBlockDefault(current_channel, time_embed_dim, ))

        self.m_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_sequence_ordering = []

        #################
        # output layers #
        #################
        for level_idx, (mult, sdim) in list(enumerate(zip(channel_mult, second_dim)))[::-1]:
            for _ in range(self.num_noattn_blocks[level_idx] + 1):
                self.layer_sequence_ordering.append('load_hidden_feature')
                extra_channel = input_block_channels.pop()
                layer = FCBlockDefault(
                    [current_channel[0] + extra_channel[0]] + current_channel[1:],
                    time_embed_dim,
                    out_channels = [mult*model_channels, sdim, 1], )

                self.add_data_layer(layer)
                current_channel = [mult*model_channels, sdim, 1]

                if with_attn[level_idx]:
                    d_head, n_heads = self.get_d_head_n_heads(current_channel[0])
                    layer = CrossAttnDefault(
                        in_channels=current_channel[0], d_head=d_head, n_heads=n_heads)
                    self.add_context_layer(layer)

            if level_idx != 0:
                layer = Linear_MultiDim(current_channel, current_channel, bias=True, )
                self.add_data_layer(layer)

        layer = nn.Sequential(
            normalization(current_channel[0]),
            nn.SiLU(),
            zero_module(Linear_MultiDim(current_channel, [output_channels], bias=True, )),
        )
        
        self.add_data_layer(layer)

        self.o_order = copy.deepcopy(self.layer_sequence_ordering)
        self.layer_order = copy.deepcopy(self.i_order + self.m_order + self.o_order)
        del self.layer_sequence_ordering

        self.parameter_group = {}
        if self.glayer_included:
            self.parameter_group['global'] = self.time_embed
        if self.dlayer_included:
            self.parameter_group['data'] = self.data_blocks
        if self.clayer_included:
            self.parameter_group['context'] = self.context_blocks

# Implementation of the Dual Guided model part of the Versatile Diffusion framework.
class UNetModelVD_DualGuided(nn.Module):
    def __init__(self):

        super().__init__()
        self.unet_image = UNetModel2D_Next().half()
        self.unet_text = UNetModel0D_Next().half()
        self.models = {'image': self.unet_image, 'text':self.unet_text}
        # self.time_embed = self.unet_image.time_embed
        # del self.unet_image.time_embed
        # del self.unet_text.time_embed
        # self.model_channels = self.unet_image.model_channels
        
    
    # This function implements the mixing methods used by the Versatile Diffusion model specifically for the Dual Guided task.
    def context_mixing(self, x, emb, context_module_list, context_info_list, mixing_type):
        nm = len(context_module_list)
        nc = len(context_info_list)
        assert nm == nc
        context = [c_info['context'] for c_info in context_info_list]
        cratio = np.array([c_info['ratio'] for c_info in context_info_list])
        # Normalize the ratio values so they sum to 1
        cratio = cratio / cratio.sum()

        if mixing_type == 'attention':
            h = None
            for module, c, r in zip(context_module_list, context, cratio):
                hi = module(x, emb, c) * r
                h = h+hi if h is not None else hi
            return h
        elif mixing_type == 'layer':
            ni = np.random.choice(nm, p=cratio)
            module = context_module_list[ni]
            c = context[ni]
            h = module(x, emb, c)
            return h


    # Forward pass 
    """
        Args:
            x (Tensor, batch_size*size):
                Latent representation of noise.
            timesteps (tensor or scalar, batch_size*1):
                Applied timesteps. 
            xtype (Str):
                Type of output data - 'image' or 'text'. Note: current model weights consider only 'image' for the dual guided model generating images.
            c0 (Tensor):
                First context, CLIP embedding format.
            c1 (Tensor):
                Second context, CLIP embedding format.
            
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            !!!Note: For the dual guided model, only one text and one image context are used!!!
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            c0_type (Str):
                Context type for c0 - 'image' or 'text'.
            c1_type (Str):
                Context type for c1 - 'image' or 'text'.
            c0_ratio (Int):
                Weight associated with c0.
            c1_ratio (Int):
                Weight associated with c1.
            mixing_type (Str):
                Type of mixing applied - 'attention' or 'layer'.
    """
    
    def forward(self, x, timesteps, xtype, c0, c1, c0_type, c1_type, c0_ratio=0.5, c1_ratio=0.5, mixing_type="attention"):

        # Convert the context information into a comprehensive dictionary
        c_info_list = [{"type": ci, "context": c, "ratio": r} for ci, c, r in zip([c0_type, c1_type], [c0, c1], [c0_ratio, c1_ratio])]
        hs = []
        model_channels = self.models[xtype].model_channels
        t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False).to(x.dtype)
        # Currently, time embeddings are sourced from the image_unet since xtype is set to 'image'.
        emb = self.models[xtype].time_embed(t_emb)

        # Initialize the data layers tailored to the target output.
        d_iter = iter(self.models[xtype].data_blocks)

       # Initialize the context layers tailored to the specific context types as a list (dual). 
        c_iter_list = [iter(self.models[c_info['type']].context_blocks) for c_info in c_info_list]
        

        i_order = self.models[xtype].i_order
        m_order = self.models[xtype].m_order
        o_order = self.models[xtype].o_order


        h = x

        for ltype in i_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module_list = [next(c_iteri) for c_iteri in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)
            elif ltype == 'save_hidden_feature':
                hs.append(h)

        for ltype in m_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module_list = [next(c_iteri) for c_iteri in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)

        for ltype in o_order:
            if ltype == 'load_hidden_feature':
                h = th.cat([h, hs.pop()], dim=1)
            elif ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module_list = [next(c_iteri) for c_iteri in c_iter_list]
                h = self.context_mixing(h, emb, module_list, c_info_list, mixing_type)
        o = h
        return o