import pytest

from tests.videos.helpers import video_segmentation, video_segmentation_alt


def test_memory_leak():
    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_video_predictor

    video_path = "../assets/act_reference_v001.mp4"

    def get_bpe_vocab_path() -> str:
        """seems missing in the sam3 package, so we add it here"""
        return hf_hub_download(
            repo_id="LanguageBind/LanguageBind", filename="open_clip/bpe_simple_vocab_16e6.txt.gz", repo_type="space"
        )

    # First run WITH video processing
    predictor1 = build_sam3_video_predictor(bpe_path=get_bpe_vocab_path())
    response = predictor1.handle_request(dict(type="start_session", resource_path=video_path))
    session_id = response["session_id"]
    predictor1.handle_request(dict(type="add_prompt", session_id=session_id, frame_index=0, text="person"))
    predictor1.handle_request(dict(type="close_session", session_id=session_id))
    predictor1.shutdown()
    del predictor1
    torch.cuda.empty_cache()

    print(f"After first predictor: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    # Second run
    predictor2 = build_sam3_video_predictor(bpe_path=get_bpe_vocab_path())
    response = predictor2.handle_request(dict(type="start_session", resource_path=video_path))
    session_id = response["session_id"]
    predictor2.handle_request(dict(type="add_prompt", session_id=session_id, frame_index=0, text="person"))
    predictor2.handle_request(dict(type="close_session", session_id=session_id))
    predictor2.shutdown()
    del predictor2
    torch.cuda.empty_cache()

    print(f"After second predictor: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")


@pytest.mark.parametrize("model", ["sam-3"])
def test_video_segmentation(model):
    video_segmentation(model)


@pytest.mark.parametrize("model", ["sam-3"])
def test_video_segmentation_alt(model):
    video_segmentation_alt(model)
