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


def test_pipeline_manual_offload_is_idempotent_and_syncs_only_explicit_boundaries():
    common = read_source("models/pipelines/_pipeline_common.py")

    assert "def _module_device" in common
    assert "current_device == target_device" in common
    assert "was_on_cuda" in common
    assert "def _manual_offload_barrier" in common
    assert "dist.is_initialized()" in common

    for relative_path in (
        "models/pipelines/pipeline_dmd_keyframe.py",
        "models/pipelines/pipeline_ref_keyframe.py",
        "models/pipelines/pipeline_pcd_keyframe.py",
    ):
        source = read_source(relative_path)
        assert "_manual_offload_barrier(\"after_text_encode\")" in source
        assert "_manual_offload_barrier(\"before_denoise\")" in source
        assert "_manual_offload_barrier(\"after_vae_decode\")" in source


def test_camera_selector_uses_batched_dinov2_and_optional_disk_cache():
    retrieval = read_source("src/retrieval_wm.py")
    video_gen = read_source("video_gen.py")

    assert "cache_dinov2_features" in retrieval
    assert ".dinov2_cache" in retrieval
    assert "def _feature_cache_key" in retrieval
    assert "self.processor(images=pil_images, return_tensors=\"pt\")" in retrieval
    assert "pooler_output.cpu().numpy()" in retrieval
    assert "del inputs, outputs" in retrieval
    assert "--cache_dinov2_features" in video_gen
    assert "cache_dinov2_features=args.cache_dinov2_features" in video_gen


def test_lazy_aux_init_synchronises_distributed_ranks():
    retrieval = read_source("src/retrieval_wm.py")

    assert "def _distributed_barrier" in retrieval
    assert "dist.is_initialized()" in retrieval
    assert "dist.get_world_size() > 1" in retrieval
    assert "self._distributed_barrier(\"before_aux_init\")" in retrieval
    assert "self._distributed_barrier(\"after_aux_init\")" in retrieval


def test_video_validation_uses_quick_probe_per_rank_vram_and_tolerant_range_check():
    video_gen = read_source("video_gen.py")
    general_utils = read_source("src/general_utils.py")

    assert "mp4_quick_check" in general_utils
    assert "from src.general_utils import set_seed, load_video, rank0_log, Timer, mp4_quick_check" in video_gen
    assert "if not mp4_quick_check(path):" in video_gen
    assert "torch.cuda.current_device()" in video_gen
    assert "torch.cuda.memory_allocated(device=cuda_device)" in video_gen
    assert "print(f\"[VRAM rank{rank}]" in video_gen
    assert "TENSOR_RANGE_EPSILON" in video_gen
    assert "raise ValueError(f\"Unexpected video tensor range" in video_gen
