import diffusers.models.transformers.transformer_qwenimage as qwen_image_module
import torch
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope


def apply_qwen_image_patches():
    """
    Monkey patch for nunchaku compatibility with diffusers.
    See: https://github.com/huggingface/diffusers/pull/12702
    """

    original_qwen_rope_forward = QwenEmbedRope.forward

    def patched_qwen_rope_forward(self, video_fhw, txt_seq_lens=None, device=None, max_txt_seq_len=None):
        # Use a safe large default (e.g. 4096) to ensure we generate enough frequencies.
        # We will slice them down in apply_rotary_emb_qwen.
        if max_txt_seq_len is None and txt_seq_lens is None:
            max_txt_seq_len = 4096
        return original_qwen_rope_forward(self, video_fhw, txt_seq_lens, device, max_txt_seq_len)  # type: ignore

    QwenEmbedRope.forward = patched_qwen_rope_forward

    # Also patch apply_rotary_emb_qwen to handle shape mismatch via slicing
    original_apply_rotary_emb_qwen = qwen_image_module.apply_rotary_emb_qwen

    def patched_apply_rotary_emb_qwen(x, freqs_cis, use_real=True, use_real_unbind_dim=-1):
        # x: [Batch, Seq, Heads, Dim]
        # freqs_cis: Tensor or Tuple
        seq_len = x.shape[1]

        # Handle the specific case crashing in nunchaku/diffusers interaction (use_real=False, tensor freqs)
        if not use_real and isinstance(freqs_cis, torch.Tensor):
            # freqs_cis typically [1, MaxSeq, Dim] or [MaxSeq, Dim]
            if freqs_cis.dim() == 3 and freqs_cis.shape[1] > seq_len:
                freqs_cis = freqs_cis[:, :seq_len]
            elif freqs_cis.dim() == 2 and freqs_cis.shape[0] > seq_len:
                freqs_cis = freqs_cis[:seq_len]

        return original_apply_rotary_emb_qwen(x, freqs_cis, use_real, use_real_unbind_dim)

    qwen_image_module.apply_rotary_emb_qwen = patched_apply_rotary_emb_qwen
