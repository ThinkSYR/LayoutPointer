# coding=utf-8
from transformers.models.bert.configuration_bert import BertConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
}


class GPLayoutLMv3Config(BertConfig):
    model_type = "gplayoutlmv3"

    def __init__(
        self,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_2d_position_embeddings=1024,
        coordinate_size=None,
        shape_size=None,
        has_relative_attention_bias=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        has_spatial_attention_bias=False,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        visual_embed=True,
        mim=False,
        wpa_task=False,
        discrete_vae_weight_path='',
        discrete_vae_type='dall-e',
        input_size=224,
        second_input_size=112,
        device='cuda',
        label_emb_size=144,
        sca_rope=2,
        sca_add_slm=True,
        pointer_rope=False,
        pointer_scale=True,
        pointer_link_heads=1,
        pointer_cls_heads=3,
        pointer_sca_layer=2,
        pointer_with_sca=True,
        using_gp=False,
        pointer_greedy=False,
        **kwargs
    ):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.max_rel_2d_pos = max_rel_2d_pos
        self.visual_embed = visual_embed
        self.mim = mim
        self.wpa_task = wpa_task
        self.discrete_vae_weight_path = discrete_vae_weight_path
        self.discrete_vae_type = discrete_vae_type
        self.input_size = input_size
        self.second_input_size = second_input_size
        self.device = device
        label_emb_size = 144 if self.hidden_size == 768 else 128
        self.label_emb_size = label_emb_size
        self.sca_rope = sca_rope
        self.sca_add_slm = sca_add_slm
        self.pointer_rope = pointer_rope
        self.pointer_scale = pointer_scale
        self.pointer_link_heads = pointer_link_heads
        self.pointer_cls_heads = pointer_cls_heads
        self.pointer_with_sca = pointer_with_sca
        self.using_gp = using_gp
        self.pointer_greedy = pointer_greedy
        self.pointer_sca_layer = pointer_sca_layer
