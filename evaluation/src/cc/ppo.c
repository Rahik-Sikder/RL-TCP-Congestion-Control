#include "ppo.h"
#include "../transport/queue.h"
/* Homebrew macOS: onnxruntime/onnxruntime_c_api.h
 * Linux pip:      onnxruntime/core/session/onnxruntime_c_api.h */
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DEBUG
#define DBG(...) fprintf(stderr, "[ppo] " __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

#define MAX_PATH 512

static const OrtApi *g_ort = NULL;

typedef struct {
    OrtEnv            *env;
    OrtSession        *session;
    OrtSessionOptions *opts;
    OrtMemoryInfo     *mem_info;
    queue_t           *queue;
    int                k;
    float              current_cwnd;  /* bytes */
    int                inference_busy;
} ppo_ctx_t;

/* Read k from {model_dir}/ppo.info.json */
static int read_k(const char *model_dir) {
    char path[MAX_PATH];
    snprintf(path, sizeof(path), "%s/ppo.info.json", model_dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[ppo] cannot open %s\n", path);
        return -1;
    }
    char buf[256];
    int k = -1;
    while (fgets(buf, sizeof(buf), f)) {
        char *p = strstr(buf, "\"k\"");
        if (p) {
            p = strchr(p, ':');
            if (p) { k = atoi(p + 1); break; }
        }
    }
    fclose(f);
    DBG("loaded k=%d from %s\n", k, path);
    return k;
}

static void ppo_run_inference(ppo_ctx_t *ctx) {
    int state_dim = ctx->k * 3 + 1;

    float      *state   = calloc((size_t)state_dim, sizeof(float));
    tcp_info_t *entries = calloc((size_t)ctx->k,    sizeof(tcp_info_t));
    if (!state || !entries) {
        fprintf(stderr, "[ppo] OOM in inference\n");
        free(state); free(entries);
        return;
    }

    int count = queue_read_all(ctx->queue, entries);
    DBG("building state vector: %d/%d entries\n", count, ctx->k);

    for (int i = 0; i < ctx->k; i++) {
        if (i < count) {
            state[i]            = entries[i].rtt_ms;
            state[ctx->k + i]   = (float)entries[i].dup_ack;
            state[2 * ctx->k + i] = (float)entries[i].timeout;
        }
        /* else: already 0.0f from calloc */
    }
    state[3 * ctx->k] = ctx->current_cwnd / (float)MSS;

    DBG("state[rtt 0..2]: %.2f %.2f %.2f  cwnd_seg=%.3f\n",
        state[0], state[1], state[2], state[3 * ctx->k]);

    /* ---- ONNX Runtime inference ---- */
    int64_t   shape[]        = {1, state_dim};
    OrtValue *input_tensor   = NULL;
    OrtValue *outputs[3]     = {NULL, NULL, NULL};
    OrtStatus *status        = NULL;

    status = g_ort->CreateTensorWithDataAsOrtValue(
        ctx->mem_info, state, (size_t)state_dim * sizeof(float),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status) {
        fprintf(stderr, "[ppo] ORT error: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        goto cleanup;
    }

    const char *input_names[]  = {"state"};
    const char *output_names[] = {"action_mean", "action_std", "value"};

    status = g_ort->Run(ctx->session, NULL,
                        input_names,
                        (const OrtValue *const *)&input_tensor, 1,
                        output_names, 3, outputs);
    if (status) {
        fprintf(stderr, "[ppo] ORT error: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        goto cleanup;
    }

    float *action_mean_data = NULL;
    status = g_ort->GetTensorMutableData(outputs[0], (void **)&action_mean_data);
    if (status) {
        fprintf(stderr, "[ppo] ORT error: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        goto cleanup;
    }

    float action_mean = action_mean_data[0];
    float new_cwnd    = ctx->current_cwnd * powf(2.0f, action_mean);
    new_cwnd = fmaxf((float)MSS, fminf(1000.0f * (float)MSS, new_cwnd));

    DBG("action_mean=%.4f old_cwnd=%.0f new_cwnd=%.0f\n",
        action_mean, ctx->current_cwnd, new_cwnd);

    ctx->current_cwnd = new_cwnd;

cleanup:
    for (int i = 0; i < 3; i++)
        if (outputs[i]) g_ort->ReleaseValue(outputs[i]);
    if (input_tensor) g_ort->ReleaseValue(input_tensor);
    free(state);
    free(entries);
}

static void ppo_on_ack(void *raw_ctx, tcp_info_t *info) {
    ppo_ctx_t *ctx = (ppo_ctx_t *)raw_ctx;

    DBG("on_ack: rtt=%.2fms dup=%d timeout=%d cwnd=%.0f\n",
        info->rtt_ms, info->dup_ack, info->timeout, info->cwnd);

    queue_push(ctx->queue, info);

    if (ctx->inference_busy) {
        DBG("inference in progress, dropping packet for inference\n");
        return;
    }
    if (queue_size(ctx->queue) < ctx->k) {
        DBG("queue not full yet (%d/%d), skipping inference\n",
            queue_size(ctx->queue), ctx->k);
        return;
    }

    ctx->inference_busy = 1;
    ppo_run_inference(ctx);
    ctx->inference_busy = 0;
}

static void ppo_on_timeout(void *raw_ctx, tcp_info_t *info) {
    ppo_on_ack(raw_ctx, info);
}

static float ppo_get_cwnd(void *raw_ctx) {
    return ((ppo_ctx_t *)raw_ctx)->current_cwnd;
}

static void ppo_destroy(void *raw_ctx) {
    ppo_ctx_t *ctx = (ppo_ctx_t *)raw_ctx;
    queue_destroy(ctx->queue);
    if (ctx->mem_info) g_ort->ReleaseMemoryInfo(ctx->mem_info);
    if (ctx->session)  g_ort->ReleaseSession(ctx->session);
    if (ctx->opts)     g_ort->ReleaseSessionOptions(ctx->opts);
    if (ctx->env)      g_ort->ReleaseEnv(ctx->env);
    free(ctx);
}

static cc_ops_t ppo_ops = {
    .on_ack     = ppo_on_ack,
    .on_timeout = ppo_on_timeout,
    .get_cwnd   = ppo_get_cwnd,
    .destroy    = ppo_destroy,
};

cc_ops_t *ppo_create(void **ctx_out, const char *model_dir) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "[ppo] failed to get ONNX Runtime API\n");
        return NULL;
    }
    DBG("ONNX Runtime API version %u loaded\n", ORT_API_VERSION);

    int k = read_k(model_dir);
    if (k <= 0) {
        fprintf(stderr, "[ppo] invalid k=%d from %s\n", k, model_dir);
        return NULL;
    }

    ppo_ctx_t *ctx    = calloc(1, sizeof(ppo_ctx_t));
    ctx->k            = k;
    ctx->current_cwnd = (float)(INIT_CWND_SEGS * MSS);
    ctx->queue        = queue_create(k);

    OrtStatus *s_env  = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ppo_eval", &ctx->env);
    if (s_env) { g_ort->ReleaseStatus(s_env); }  /* non-fatal: env failures are recoverable */
    OrtStatus *s_opts = g_ort->CreateSessionOptions(&ctx->opts);
    if (s_opts) { g_ort->ReleaseStatus(s_opts); }

    char onnx_path[MAX_PATH];
    snprintf(onnx_path, sizeof(onnx_path), "%s/ppo.onnx", model_dir);
    DBG("loading model from %s\n", onnx_path);

    OrtStatus *status = g_ort->CreateSession(ctx->env, onnx_path,
                                             ctx->opts, &ctx->session);
    if (status) {
        fprintf(stderr, "[ppo] CreateSession failed: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        queue_destroy(ctx->queue);
        g_ort->ReleaseSessionOptions(ctx->opts);
        g_ort->ReleaseEnv(ctx->env);
        free(ctx);
        return NULL;
    }

    OrtStatus *s_mem = g_ort->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0,
                                               OrtMemTypeDefault, &ctx->mem_info);
    if (s_mem) { g_ort->ReleaseStatus(s_mem); }

    DBG("model loaded. k=%d state_dim=%d initial_cwnd=%.0f\n",
        k, k * 3 + 1, ctx->current_cwnd);

    *ctx_out = ctx;
    return &ppo_ops;
}
