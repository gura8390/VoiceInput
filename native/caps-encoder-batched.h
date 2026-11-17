#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(_WIN32)
#define CAPS_ENCODER_API __declspec(dllexport)
#else
#define CAPS_ENCODER_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
typedef struct caps_ggml_encoder_stage_metrics {
    uint64_t gpu_core, cpu_core, gpu_metadata, cpu_metadata, unknown;
    uint64_t graph_calls, gpu_compute_bytes, cpu_compute_bytes;
    double elapsed_seconds;
} caps_ggml_encoder_stage_metrics;
CAPS_ENCODER_API int32_t caps_ggml_encoder_get_stage_metrics(void * handle, int32_t stage, caps_ggml_encoder_stage_metrics * out);
CAPS_ENCODER_API uint32_t caps_ggml_encoder_abi_version(void);
CAPS_ENCODER_API void * caps_ggml_encoder_load(const char * mmproj, const char * device_name, int32_t threads, int32_t use_gpu);
CAPS_ENCODER_API int32_t caps_ggml_encoder_dimension(void * handle);
CAPS_ENCODER_API size_t caps_ggml_encoder_output_rows(size_t frames);
CAPS_ENCODER_API int32_t caps_ggml_encoder_encode(void * handle, const float * mel, size_t frames, size_t valid_rows, float * out, size_t capacity_floats);
CAPS_ENCODER_API const char * caps_ggml_encoder_backend_name(void * handle);
CAPS_ENCODER_API const char * caps_ggml_encoder_last_error(void);
CAPS_ENCODER_API int32_t caps_ggml_encoder_stats(void * handle, uint64_t * gpu_nodes, uint64_t * cpu_nodes, uint64_t * total_nodes);
CAPS_ENCODER_API int32_t caps_ggml_encoder_stats_detail(void * handle, uint64_t * gpu_core, uint64_t * cpu_core, uint64_t * gpu_metadata, uint64_t * cpu_metadata, uint64_t * unknown);
CAPS_ENCODER_API void caps_ggml_encoder_free(void * handle);
#ifdef __cplusplus
}
#endif
