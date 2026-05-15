// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// diffusionfile.cpp — chimerafile diffusion backend
//
// Wraps the stable-diffusion.cpp C API (txt2img / img2img) so that
//
//   ./chimerafile diffusion [options...]
//
// works with the same CLI conventions as the other engines.
//
// The C API lives in llamafile's stable-diffusion.cpp submodule; its
// sources are compiled against llama.cpp's GGML (single-GGML model),
// and stb implementations come from third_party/stb.

#include <cosmo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <string>
#include <vector>

#include "llamafile/llamafile.h"

// stable-diffusion C API
#include "stable-diffusion.h"

// GGML backend API (for device enumeration / diagnostics)
#include "ggml-backend.h"

// stb for image I/O (declarations only; implementations are in stb.a)
#include "stb_image.h"
#include "stb_image_write.h"

// LLAMAFILE_VERSION_STRING is defined by BUILD.mk
#ifndef LLAMAFILE_VERSION_STRING
#define LLAMAFILE_VERSION_STRING "0.0.0-dev"
#endif

// ── Defaults ────────────────────────────────────────────────────────────────

static const int    DEFAULT_WIDTH     = 512;
static const int    DEFAULT_HEIGHT    = 512;
static const int    DEFAULT_STEPS     = 20;
static const float  DEFAULT_CFG_SCALE = 7.0f;
static const int    DEFAULT_SEED      = 42;
static const char   DEFAULT_OUTPUT[]  = "output.png";

// ─── CLI argument parsing (minimal; covers txt2img) ─────────────────────────

struct DiffusionParams {
    // model / paths
    std::string model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string controlnet_path;
    std::string lora_model_dir;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;
    std::string input_id_images_path;
    std::string output_path = DEFAULT_OUTPUT;
    std::string input_path;               // img2img input
    std::string control_image_path;
    std::string prompt;
    std::string negative_prompt;

    // generation params
    int width           = DEFAULT_WIDTH;
    int height          = DEFAULT_HEIGHT;
    int sample_steps    = DEFAULT_STEPS;
    float cfg_scale     = DEFAULT_CFG_SCALE;
    float strength      = 0.75f;          // img2img only
    float control_strength = 0.5f;
    float style_ratio   = 0.0f;
    int clip_skip       = -1;             // -1 = auto
    int batch_count     = 1;
    int64_t seed        = DEFAULT_SEED;

    // back-end flags
    enum sample_method_t sample_method = EULER_A;
    enum schedule_t      schedule      = DEFAULT;
    enum rng_type_t      rng_type      = STD_DEFAULT_RNG;
    enum sd_type_t       wtype         = SD_TYPE_F32;

    // booleans
    bool verbose         = false;
    bool vae_tiling      = false;
    bool vae_decode_only = true;
    bool free_params     = true;
    bool normalize_input = false;
    bool clip_on_cpu     = false;
    bool control_net_cpu = false;
    bool vae_on_cpu      = false;
    bool canny_preprocess = false;
    bool img2img_mode    = false;
    int  upscale_repeats = 0;
};

static void print_usage(const char *prog) {
    printf(
        "chimerafile: diffusion engine — text-to-image & image-to-image\n"
        "(based on stable-diffusion.cpp)\n"
        "\n"
        "usage: %s diffusion [options]\n"
        "\n"
        "required:\n"
        "  -m, --model FILE          path to model (.safetensors / .ckpt / .gguf)\n"
        "  -p, --prompt TEXT         text prompt\n"
        "\n"
        "optional:\n"
        "  --neg-prompt TEXT         negative prompt\n"
        "  --cfg-scale F             classifier-free guidance scale (default: %.1f)\n"
        "  --steps N                 number of sampling steps (default: %d)\n"
        "  -W, --width N             image width (default: %d)\n"
        "  -H, --height N            image height (default: %d)\n"
        "  --seed N                  random seed (default: %d)\n"
        "  -o, --output FILE         output image path (default: %s)\n"
        "  -b, --batch N             batch count (default: %d)\n"
        "  --sampler METHOD          euler_a, euler, heun, dpm2, dpm++2m, lcm\n"
        "                            (default: euler_a)\n"
        "  --schedule TYPE           default, discrete, karras, ays\n"
        "  --strength F              img2img strength (default: %.2f)\n"
        "  --input FILE              input image for img2img\n"
        "\n"
        "  --vae FILE                separate VAE model\n"
        "  --taesd FILE              tiny auto-encoder for fast preview\n"
        "  --esrgan FILE             ESRGAN upscaler model\n"
        "  --upscale N               run upscaler N times on output\n"
        "  --controlnet FILE         ControlNet model\n"
        "  --control-image FILE      control image for ControlNet\n"
        "  --canny                   apply Canny preprocessor to control image\n"
        "\n"
        "  --lora-dir DIR            LoRA model directory\n"
        "  --embeddings DIR          embeddings directory\n"
        "\n"
        "  --vae-tiling              enable VAE tiling (reduces VRAM)\n"
        "  --rng TYPE                std_default, cuda\n"
        "  --wtype TYPE              weight type (f32, f16, q4_0, ...)\n"
        "  --verbose                 verbose logging\n"
        "  --version                 show version\n"
        "\n"
        "examples:\n"
        "  %s diffusion -m model.safetensors -p \"a cat in a spacesuit\"\n"
        "  %s diffusion -m model.safetensors -p \"portrait\" -W 1024 -H 1024 --steps 30\n"
        "  %s diffusion -m model.safetensors -p \"turn into van gogh\" \\\n"
        "       --input photo.jpg --strength 0.6\n",
        prog,
        DEFAULT_CFG_SCALE, DEFAULT_STEPS,
        DEFAULT_WIDTH, DEFAULT_HEIGHT,
        DEFAULT_SEED, DEFAULT_OUTPUT, 1,
        0.75f,
        prog, prog, prog);
}

static bool parse_args(int argc, char **argv, DiffusionParams &p) {
    // We expect: argv[0] = "chimerafile" (shifted already), remaining args follow
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            return false; // caller will print usage
        }

        // ── required ────────────────────────────────────────────────
        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) { fprintf(stderr, "error: --model requires a path\n"); return false; }
            p.model_path = argv[i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (++i >= argc) { fprintf(stderr, "error: --prompt requires text\n"); return false; }
            p.prompt = argv[i];
        } else if (arg == "--neg-prompt" || arg == "--negative-prompt") {
            if (++i >= argc) { fprintf(stderr, "error: --neg-prompt requires text\n"); return false; }
            p.negative_prompt = argv[i];

        // ── image geometry ──────────────────────────────────────────
        } else if (arg == "-W" || arg == "--width") {
            if (++i >= argc) { fprintf(stderr, "error: --width requires N\n"); return false; }
            p.width = atoi(argv[i]);
        } else if (arg == "-H" || arg == "--height") {
            if (++i >= argc) { fprintf(stderr, "error: --height requires N\n"); return false; }
            p.height = atoi(argv[i]);

        // ── sampling ────────────────────────────────────────────────
        } else if (arg == "--steps" || arg == "-s") {
            if (++i >= argc) { fprintf(stderr, "error: --steps requires N\n"); return false; }
            p.sample_steps = atoi(argv[i]);
        } else if (arg == "--cfg-scale" || arg == "--cfg") {
            if (++i >= argc) { fprintf(stderr, "error: --cfg-scale requires F\n"); return false; }
            p.cfg_scale = atof(argv[i]);
        } else if (arg == "--seed") {
            if (++i >= argc) { fprintf(stderr, "error: --seed requires N\n"); return false; }
            p.seed = atoll(argv[i]);
        } else if (arg == "--sampler" || arg == "--sample-method") {
            if (++i >= argc) { fprintf(stderr, "error: --sampler requires name\n"); return false; }
            std::string s = argv[i];
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            if (s == "euler_a")        p.sample_method = EULER_A;
            else if (s == "euler")     p.sample_method = EULER;
            else if (s == "heun")      p.sample_method = HEUN;
            else if (s == "dpm2")      p.sample_method = DPM2;
            else if (s == "dpm++2m" || s == "dpmpp2m") p.sample_method = DPMPP2M;
            else if (s == "lcm")       p.sample_method = LCM;
            else { fprintf(stderr, "error: unknown sampler '%s'\n", argv[i]); return false; }
        } else if (arg == "--schedule") {
            if (++i >= argc) { fprintf(stderr, "error: --schedule requires name\n"); return false; }
            std::string s = argv[i];
            if (s == "default")         p.schedule = DEFAULT;
            else if (s == "discrete")   p.schedule = DISCRETE;
            else if (s == "karras")     p.schedule = KARRAS;
            else if (s == "ays")        p.schedule = AYS;
            else { fprintf(stderr, "error: unknown schedule '%s'\n", argv[i]); return false; }

        // ── I/O ─────────────────────────────────────────────────────
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) { fprintf(stderr, "error: --output requires path\n"); return false; }
            p.output_path = argv[i];
        } else if (arg == "--input" || arg == "-i") {
            if (++i >= argc) { fprintf(stderr, "error: --input requires path\n"); return false; }
            p.input_path = argv[i];
            p.img2img_mode = true;
        } else if (arg == "-b" || arg == "--batch") {
            if (++i >= argc) { fprintf(stderr, "error: --batch requires N\n"); return false; }
            p.batch_count = std::max(1, atoi(argv[i]));

        // ── models ──────────────────────────────────────────────────
        } else if (arg == "--vae") {
            if (++i >= argc) { fprintf(stderr, "error: --vae requires path\n"); return false; }
            p.vae_path = argv[i];
            p.vae_decode_only = true;
        } else if (arg == "--taesd") {
            if (++i >= argc) { fprintf(stderr, "error: --taesd requires path\n"); return false; }
            p.taesd_path = argv[i];
        } else if (arg == "--esrgan") {
            if (++i >= argc) { fprintf(stderr, "error: --esrgan requires path\n"); return false; }
            p.esrgan_path = argv[i];
        } else if (arg == "--upscale") {
            if (++i >= argc) { fprintf(stderr, "error: --upscale requires N\n"); return false; }
            p.upscale_repeats = atoi(argv[i]);
        } else if (arg == "--controlnet") {
            if (++i >= argc) { fprintf(stderr, "error: --controlnet requires path\n"); return false; }
            p.controlnet_path = argv[i];
        } else if (arg == "--control-image") {
            if (++i >= argc) { fprintf(stderr, "error: --control-image requires path\n"); return false; }
            p.control_image_path = argv[i];
        } else if (arg == "--canny") {
            p.canny_preprocess = true;
        } else if (arg == "--lora-dir") {
            if (++i >= argc) { fprintf(stderr, "error: --lora-dir requires path\n"); return false; }
            p.lora_model_dir = argv[i];
        } else if (arg == "--embeddings") {
            if (++i >= argc) { fprintf(stderr, "error: --embeddings requires path\n"); return false; }
            p.embeddings_path = argv[i];

        // ── flags ───────────────────────────────────────────────────
        } else if (arg == "--vae-tiling") {
            p.vae_tiling = true;
        } else if (arg == "--rng") {
            if (++i >= argc) { fprintf(stderr, "error: --rng requires type\n"); return false; }
            if (strcmp(argv[i], "cuda") == 0) p.rng_type = CUDA_RNG;
            else p.rng_type = STD_DEFAULT_RNG;
        } else if (arg == "--wtype") {
            if (++i >= argc) { fprintf(stderr, "error: --wtype requires type\n"); return false; }
            std::string t = argv[i];
            std::transform(t.begin(), t.end(), t.begin(), ::toupper);
            if (t == "F32")      p.wtype = SD_TYPE_F32;
            else if (t == "F16") p.wtype = SD_TYPE_F16;
            else if (t == "Q4_0") p.wtype = SD_TYPE_Q4_0;
            else if (t == "Q8_0") p.wtype = SD_TYPE_Q8_0;
            else if (t == "Q4_K") p.wtype = SD_TYPE_Q4_K;
            else { fprintf(stderr, "error: unsupported wtype '%s'\n", argv[i]); return false; }
        } else if (arg == "--verbose") {
            p.verbose = true;
        } else if (arg == "--version") {
            puts("chimerafile diffusion v" LLAMAFILE_VERSION_STRING);
            exit(0);

        // ── unknown ─────────────────────────────────────────────────
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", arg.c_str());
            return false;
        }
    }

    if (p.model_path.empty()) {
        fprintf(stderr, "error: -m/--model is required\n");
        return false;
    }
    if (p.prompt.empty()) {
        fprintf(stderr, "error: -p/--prompt is required\n");
        return false;
    }
    return true;
}

// ── Log callback (passthrough to llamafile's null callback if not verbose) ──

static void sd_log_cb(enum sd_log_level_t level, const char *text, void *data) {
    bool *verbose = (bool *)data;
    if (!*verbose && level > SD_LOG_WARN)
        return;
    // forward to stderr; stable-diffusion.cpp messages often end with \n
    fputs(text, stderr);
}

// ── Progress callback ───────────────────────────────────────────────────────

static void progress_cb(int step, int steps, float time, void *data) {
    (void)data;
    if (steps > 0) {
        int pct = (step * 100) / steps;
        fprintf(stderr, "\r  step %d/%d (%d%%)  [", step, steps, pct);
        int bar = 30;
        int pos = (step * bar) / steps;
        for (int i = 0; i < bar; i++)
            fputc(i < pos ? '=' : ' ', stderr);
        fprintf(stderr, "]  %.1fs", time);
        if (step >= steps)
            fputc('\n', stderr);
    }
}

// ── Main entry point (called from chimerafile dispatcher) ──────────────────

extern "C" int diffusionfile_main(int argc, char **argv) {
    // CPU feature check & crash reports
    llamafile_check_cpu();
    ShowCrashReports();

    // Load bundled args from zip section (for self-extracting binaries)
    argc = cosmo_args("/zip/.args", &argv);

    // Parse our options
    DiffusionParams p;
    if (!parse_args(argc, argv, p)) {
        print_usage(argv[0]);
        return 1;
    }

    // ── Detect GPU backends ────────────────────────────────────────
    // llamafile bundles GPU backends as .so files inside the APE,
    // loaded lazily at runtime.  Calling llamafile_has_gpu() triggers
    // the probe & load cycle so ggml_backend_dev_by_type() finds them.
    bool have_gpu = llamafile_has_gpu();

    // Set up logging (after gpu probe so we can report the result)
    sd_set_log_callback(sd_log_cb, &p.verbose);
    sd_set_progress_callback(progress_cb, NULL);

    if (p.verbose) {
        fprintf(stderr, "chimerafile diffusion v" LLAMAFILE_VERSION_STRING "\n");
        fprintf(stderr, "  model : %s\n", p.model_path.c_str());
        fprintf(stderr, "  size  : %dx%d\n", p.width, p.height);
        fprintf(stderr, "  steps : %d\n", p.sample_steps);
        fprintf(stderr, "  cfg   : %.1f\n", p.cfg_scale);
        fprintf(stderr, "  seed  : %ld\n", p.seed);
        if (!p.input_path.empty())
            fprintf(stderr, "  mode  : img2img  (input: %s, strength: %.2f)\n",
                    p.input_path.c_str(), p.strength);
        else
            fprintf(stderr, "  mode  : txt2img\n");
    }

    // Report available backends
    if (have_gpu) {
        size_t n = ggml_backend_dev_count();
        fprintf(stderr, "  backends:");
        for (size_t i = 0; i < n; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            struct ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            fprintf(stderr, " %s(%s)", props.name,
                    props.type == GGML_BACKEND_DEVICE_TYPE_GPU ? "GPU" :
                    props.type == GGML_BACKEND_DEVICE_TYPE_IGPU ? "IGPU" :
                    props.type == GGML_BACKEND_DEVICE_TYPE_CPU ? "CPU" :
                    props.type == GGML_BACKEND_DEVICE_TYPE_ACCEL ? "ACCEL" :
                    "?");
        }
        fprintf(stderr, "\n");
    } else {
        fprintf(stderr, "  backend: CPU (no GPU detected)\n");
    }

    // ── Load input image (img2img) ──────────────────────────────────
    uint8_t *input_image_buffer = NULL;
    uint8_t *control_image_buffer = NULL;
    int input_width = 0, input_height = 0, input_channels = 0;
    bool vae_decode_only = true;

    if (p.img2img_mode && !p.input_path.empty()) {
        vae_decode_only = false;
        input_image_buffer = stbi_load(p.input_path.c_str(),
                                       &input_width, &input_height,
                                       &input_channels, 3);
        if (!input_image_buffer) {
            fprintf(stderr, "error: failed to load input image '%s'\n",
                    p.input_path.c_str());
            return 1;
        }
        if (input_width <= 0 || input_height <= 0) {
            fprintf(stderr, "error: invalid input image dimensions\n");
            free(input_image_buffer);
            return 1;
        }
        // Resize if needed (stb does this implicitly via the API)

        // Only use vae_decode_only = false for img2img
        vae_decode_only = false;
    }

    // ── Create SD context ───────────────────────────────────────────
    // NOTE: stable-diffusion.cpp's new_sd_ctx constructs std::string from
    // each C string without NULL checks, so always pass a valid pointer.
    sd_ctx_t *sd_ctx = new_sd_ctx(
        p.model_path.c_str(),
        p.vae_path.empty() ? "" : p.vae_path.c_str(),
        p.taesd_path.empty() ? "" : p.taesd_path.c_str(),
        p.controlnet_path.empty() ? "" : p.controlnet_path.c_str(),
        p.lora_model_dir.empty() ? "" : p.lora_model_dir.c_str(),
        p.embeddings_path.empty() ? "" : p.embeddings_path.c_str(),
        p.stacked_id_embeddings_path.empty() ? "" : p.stacked_id_embeddings_path.c_str(),
        vae_decode_only,
        p.vae_tiling,
        p.free_params,
        // n_threads: use all available cores
        get_num_physical_cores(),
        p.wtype,
        p.rng_type,
        p.schedule,
        p.clip_on_cpu,
        p.control_net_cpu,
        p.vae_on_cpu);

    if (!sd_ctx) {
        fprintf(stderr, "error: failed to create SD context\n");
        free(input_image_buffer);
        return 1;
    }

    // ── Control image ───────────────────────────────────────────────
    sd_image_t *control_image = NULL;
    sd_image_t control_img_storage = {};
    if (!p.controlnet_path.empty() && !p.control_image_path.empty()) {
        int c = 0, cw = 0, ch = 0;
        control_image_buffer = stbi_load(p.control_image_path.c_str(),
                                         &cw, &ch,
                                         &c, 3);
        control_img_storage.width  = (uint32_t)cw;
        control_img_storage.height = (uint32_t)ch;
        if (!control_image_buffer) {
            fprintf(stderr, "error: failed to load control image '%s'\n",
                    p.control_image_path.c_str());
            free_sd_ctx(sd_ctx);
            free(input_image_buffer);
            return 1;
        }
        control_img_storage.channel = 3;
        control_img_storage.data = control_image_buffer;
        control_image = &control_img_storage;

        if (p.canny_preprocess) {
            control_img_storage.data =
                preprocess_canny(control_img_storage.data,
                                 control_img_storage.width,
                                 control_img_storage.height,
                                 0.08f, 0.08f, 0.8f, 1.0f, false);
        }
    }

    // ── Generate ────────────────────────────────────────────────────
    sd_image_t *results = NULL;

    if (p.img2img_mode && input_image_buffer) {
        sd_image_t input_image = {
            (uint32_t)input_width,
            (uint32_t)input_height,
            3,
            input_image_buffer
        };

        results = img2img(sd_ctx,
                          input_image,
                          p.prompt.c_str(),
                          p.negative_prompt.c_str(),
                          p.clip_skip,
                          p.cfg_scale,
                          p.width,
                          p.height,
                          p.sample_method,
                          p.sample_steps,
                          p.strength,
                          p.seed,
                          p.batch_count,
                          control_image,
                          p.control_strength,
                          p.style_ratio,
                          p.normalize_input,
                          p.input_id_images_path.empty() ? "" : p.input_id_images_path.c_str());
    } else {
        results = txt2img(sd_ctx,
                          p.prompt.c_str(),
                          p.negative_prompt.c_str(),
                          p.clip_skip,
                          p.cfg_scale,
                          p.width,
                          p.height,
                          p.sample_method,
                          p.sample_steps,
                          p.seed,
                          p.batch_count,
                          control_image,
                          p.control_strength,
                          p.style_ratio,
                          p.normalize_input,
                          p.input_id_images_path.empty() ? "" : p.input_id_images_path.c_str());
    }

    if (!results) {
        fprintf(stderr, "error: generation failed\n");
        free_sd_ctx(sd_ctx);
        free(input_image_buffer);
        free(control_image_buffer);
        return 1;
    }

    // ── Upscale ───────────────────────────────────────────────────────
    if (!p.esrgan_path.empty() && p.upscale_repeats > 0) {
        upscaler_ctx_t *upscaler_ctx = new_upscaler_ctx(p.esrgan_path.c_str(),
                                                         get_num_physical_cores(),
                                                         p.wtype);
        if (upscaler_ctx) {
            int upscale_factor = 4; // default for RealESRGAN
            for (int i = 0; i < p.batch_count; i++) {
                if (!results[i].data)
                    continue;
                sd_image_t current = results[i];
                for (int u = 0; u < p.upscale_repeats; u++) {
                    sd_image_t up = upscale(upscaler_ctx, current, upscale_factor);
                    if (!up.data) {
                        fprintf(stderr, "warning: upscale failed for image %d\n", i);
                        break;
                    }
                    free(current.data);
                    current = up;
                }
                results[i] = current;
            }
            free_upscaler_ctx(upscaler_ctx);
        } else {
            fprintf(stderr, "warning: failed to create upscaler context\n");
        }
    }

    // ── Save output ────────────────────────────────────────────────
    for (int i = 0; i < p.batch_count; i++) {
        if (!results[i].data)
            continue;

        std::string path = p.output_path;
        if (p.batch_count > 1) {
            size_t dot = path.find_last_of('.');
            std::string base = (dot == std::string::npos)
                ? path : path.substr(0, dot);
            std::string ext = (dot == std::string::npos)
                ? ".png" : path.substr(dot);
            path = base + "_" + std::to_string(i + 1) + ext;
        }

        int ok = stbi_write_png(path.c_str(),
                                (int)results[i].width,
                                (int)results[i].height,
                                (int)results[i].channel,
                                results[i].data,
                                0,
                                NULL);
        if (ok) {
            fprintf(stderr, "  saved: %s  (%dx%d)\n",
                    path.c_str(), results[i].width, results[i].height);
        } else {
            fprintf(stderr, "error: failed to write '%s'\n", path.c_str());
        }
        free(results[i].data);
    }
    free(results);

    // ── Cleanup ─────────────────────────────────────────────────────
    free_sd_ctx(sd_ctx);
    free(input_image_buffer);
    free(control_image_buffer);

    return 0;
}
