from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from collections import OrderedDict
from .outputs import BaseOutput
from model.diffusers.models.unet_2d_condition import UNet2DConditionModel

all_feature_dic={}

def clear_feature_dic():
    global all_feature_dic
    all_feature_dic={}
    all_feature_dic["low"]=[]
    all_feature_dic["mid"]=[]
    all_feature_dic["high"]=[]
    all_feature_dic["highest"]=[]
def get_feature_dic():
    global all_feature_dic
    return all_feature_dic


class UNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor


class UNet2D(UNet2DConditionModel):


    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """r
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps.to(dtype=torch.float32)
            timesteps = timesteps[None].to(device=sample.device)
        

#         if timesteps[0]==0:
#             flag_time=True
#         else:
#             flag_time=False
        
        if timesteps[0]==0 or timesteps[0]==1:
            flag_time=True
        else:
            flag_time=False    
            
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)
        
        if flag_time:
            reshape_h=sample.reshape(int(sample.size()[0]/2),int(sample.size()[1]*2),sample.size()[2],sample.size()[3])
            if reshape_h.size()[2]==8:
                all_feature_dic["low"].append(reshape_h)
            elif reshape_h.size()[2]==16:
                all_feature_dic["mid"].append(reshape_h)
            elif reshape_h.size()[2]==32:
                all_feature_dic["high"].append(reshape_h)
            elif reshape_h.size()[2]==64:
                all_feature_dic["highest"].append(reshape_h)
                    
        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples = downsample_block(
                    hidden_states=sample, temb=emb, encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            if flag_time:
                for h in res_samples:
                    reshape_h=h.reshape(int(h.size()[0]/2),int(h.size()[1]*2),h.size()[2],h.size()[3])
#                     print(reshape_h.size()[2])
                    if reshape_h.size()[2]==8:
                        all_feature_dic["low"].append(reshape_h)
                    elif reshape_h.size()[2]==16:
                        all_feature_dic["mid"].append(reshape_h)
                    elif reshape_h.size()[2]==32:
                        all_feature_dic["high"].append(reshape_h)
                    elif reshape_h.size()[2]==64:
                        all_feature_dic["highest"].append(reshape_h)
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)
        if flag_time:
            reshape_h=sample.reshape(int(sample.size()[0]/2),int(sample.size()[1]*2),sample.size()[2],sample.size()[3])
            if reshape_h.size()[2]==8:
                all_feature_dic["low"].append(reshape_h)
            elif reshape_h.size()[2]==16:
                all_feature_dic["mid"].append(reshape_h)
            elif reshape_h.size()[2]==32:
                all_feature_dic["high"].append(reshape_h)
            elif reshape_h.size()[2]==64:
                all_feature_dic["highest"].append(reshape_h)
        
        # 5. up
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample, up_samples = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, up_samples = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)
            
            if flag_time:
                for h in up_samples:
                    reshape_h=h.reshape(int(h.size()[0]/2),int(h.size()[1]*2),h.size()[2],h.size()[3])
#                     reshape_h=sample.reshape(int(sample.size()[0]/2),int(sample.size()[1]*2),sample.size()[2],sample.size()[3])
                    if reshape_h.size()[2]==8:
                        all_feature_dic["low"].append(reshape_h)
                    elif reshape_h.size()[2]==16:
                        all_feature_dic["mid"].append(reshape_h)
                    elif reshape_h.size()[2]==32:
                        all_feature_dic["high"].append(reshape_h)
                    elif reshape_h.size()[2]==64:
                        all_feature_dic["highest"].append(reshape_h)
                    
        # 6. post-process
        # make sure hidden states is in float32
        # when running in half-precision
        sample = self.conv_norm_out(sample.float()).type(sample.dtype)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        
        
        
        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)