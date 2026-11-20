import pytest
import torch


@pytest.fixture
def model_factory():
    """Random small official modules. NEVER a pretrained waveform qualification."""
    from voxcpm.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig
    from voxcpm.modules.audiovae import AudioVAEV2, AudioVAEConfigV2
    class Tokenizer:
        vocab = {"test": 5}
        def tokenize(self, text, **kwargs): return ["test"] * len(text.split())
        def convert_tokens_to_ids(self, tokens): return [5] * len(tokens)
    torch.set_num_threads(2)
    torch.manual_seed(219)
    lm = dict(bos_token_id=1, eos_token_id=2, hidden_size=16, intermediate_size=32,
              max_position_embeddings=128, num_attention_heads=2, num_hidden_layers=1,
              num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=128, use_mup=False,
              scale_emb=1., dim_model_base=16, scale_depth=1., rope_theta=10000.,
              rope_scaling=dict(type="longrope", long_factor=[1.] * 4, short_factor=[1.] * 4,
                                original_max_position_embeddings=128))
    vae_config = AudioVAEConfigV2(encoder_dim=8, encoder_rates=[2, 2], latent_dim=4,
                                 decoder_dim=32, decoder_rates=[3, 2, 2], depthwise=False,
                                 sr_bin_boundaries=None, cond_dim=8)
    config = VoxCPMConfig(lm_config=lm, patch_size=2, feat_dim=4, residual_lm_num_layers=1,
                         scalar_quantization_latent_dim=8, encoder_config=dict(
                             hidden_dim=16, ffn_dim=32, num_heads=2, num_layers=1),
                         dit_config=dict(hidden_dim=16, ffn_dim=32, num_heads=2, num_layers=1,
                                         cfm_config={}),
                         max_length=128, device="cpu", dtype="float32")
    def construct():
        return VoxCPM2Model(config, Tokenizer(), AudioVAEV2(vae_config), device="cpu").eval().requires_grad_(False)
    return construct


@pytest.fixture
def model(model_factory):
    return model_factory()
