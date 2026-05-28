from pathlib import Path


WORLDGEN_ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path: str) -> str:
    return (WORLDGEN_ROOT / relative_path).read_text(encoding="utf-8")


def test_video_gen_exposes_single_gpu_optimization_flags_and_validates_skips():
    source = read_source("video_gen.py")

    assert "--lazy_aux_models" in source
    assert "--no_compile" in source
    assert "--offload_mode" in source
    assert "--strict_skip_validation" in source
    assert "--log_vram" in source
    assert "def validate_video_file" in source
    assert "def log_cuda_memory" in source
    assert "validate_video_file(" in source


def test_worldstereo_wrapper_threads_compile_and_manual_offload_options():
    source = read_source("models/worldstereo_wrapper.py")

    assert "compile_models: bool = True" in source
    assert "offload_mode: str = \"none\"" in source
    assert "if compile_models:" in source
    assert "enable_manual_cpu_offload" in source
    assert "_apply_offload_mode" in source


def test_pipeline_manual_cpu_offload_hooks_are_present_in_all_keyframe_pipelines():
    common = read_source("models/pipelines/_pipeline_common.py")
    assert "def enable_manual_cpu_offload" in common
    assert "def _ensure_model_on_device" in common
    assert "def _offload_model_to_cpu" in common

    for relative_path in (
        "models/pipelines/pipeline_dmd_keyframe.py",
        "models/pipelines/pipeline_ref_keyframe.py",
        "models/pipelines/pipeline_pcd_keyframe.py",
    ):
        source = read_source(relative_path)
        assert "_ensure_model_on_device(\"text_encoder\"" in source
        assert "_offload_model_to_cpu(\"text_encoder\"" in source
        assert "_ensure_model_on_device(\"image_encoder\"" in source
        assert "_offload_model_to_cpu(\"image_encoder\"" in source
        assert "_ensure_model_on_device(\"vae\"" in source
        assert "_offload_model_to_cpu(\"vae\"" in source
        assert "self.maybe_free_model_hooks()" in source


def test_memory_bank_can_defer_aux_models_until_alignment():
    source = read_source("src/retrieval_wm.py")

    assert "defer_aux_models" in source
    assert "def ensure_aux_models" in source
    assert "self.ensure_aux_models(" in source
    assert "self._aux_models_deferred" in source


def test_camera_selector_and_alignment_points_are_lazy_gpu_residents():
    source = read_source("src/retrieval_wm.py")

    assert "def ensure_model" in source
    assert "def offload_model" in source
    assert "self.model = None" in source
    assert "self._load_model(feature_extractor)" not in source
    assert "self._points = None" in source
    assert "def get_points" in source
    assert "def release_points" in source
    assert "self.release_points()" in source


def test_video_gen_reuses_generated_frames_without_mp4_decode_roundtrip():
    source = read_source("video_gen.py")

    assert "def load_validated_video" in source
    assert "def tensor_video_to_pil_frames" in source
    assert "generated_frames = tensor_video_to_pil_frames(output)" in source
    assert "memory_bank.update_memory(gen_frames=generated_frames" in source
    assert "Reload results for memory update" not in source
