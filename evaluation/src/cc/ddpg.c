#include "ddpg.h"
#include "../transport/queue.h"
#include <onnxruntime_c_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DEBUG
#define DBG(...) fprintf(stderr, "[ddpg] " __VA_ARGS__)
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
    /* rolling loss-rate counters */
    long               total_packets;
    long               lost_packets;
} ddpg_ctx_t;

static int read_k(const char *model_dir) {
    char path[MAX_PATH];
    snprintf(path, sizeof(path), "%s/ddpg.info.json", model_dir);

    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[ddpg] cannot open %s\n", path);
        char ls_cmd[MAX_PATH + 8];
        snprintf(ls_cmd, sizeof(ls_cmd), "ls -la %s", model_dir);
        system(ls_cmd);
        return -1;
    }
    char buf[256];
    int k = -1;
    while (fgets(buf, sizeof(buf), f)) {
        char *p = strstr(buf, "\"k\"");
        if (p && (p = strchr(p, ':'))) { k = atoi(p + 1); break; }
    }
    fclose(f);
    DBG("loaded k=%d from %s\n", k, path);
    return k;
}

static void ddpg_run_inference(ddpg_ctx_t *ctx) {
    int state_dim = ctx->k * 3 + 2;  /* rtt, dup_ack, timeout × k + cwnd + loss_rate */

    float      *state   = calloc((size_t)state_dim, sizeof(float));
    tcp_info_t *entries = calloc((size_t)ctx->k,    sizeof(tcp_info_t));
    if (!state || !entries) {
        fprintf(stderr, "[ddpg] OOM in inference\n");
        free(state); free(entries);
        return;
    }

    int count = queue_read_all(ctx->queue, entries);
    DBG("building state vector: %d/%d entries\n", count, ctx->k);

    int loss_events = 0;
    for (int i = 0; i < ctx->k; i++) {
        if (i < count) {
            state[i]              = entries[i].rtt_ms;
            state[ctx->k + i]     = (float)entries[i].dup_ack;
            state[2 * ctx->k + i] = (float)entries[i].timeout;
            if (entries[i].dup_ack || entries[i].timeout)
                loss_events++;
        }
    }
    float cwnd_mss = ctx->current_cwnd / (float)MSS;
    state[3 * ctx->k]     = cwnd_mss;
    state[3 * ctx->k + 1] = ctx->total_packets > 0
                            ? (float)ctx->lost_packets / (float)ctx->total_packets
                            : (float)loss_events / (float)(count > 0 ? count : 1);

    DBG("state[rtt 0..2]: %.2f %.2f %.2f  cwnd_seg=%.3f loss=%.4f\n",
        state[0], state[1], state[2], state[3 * ctx->k], state[3 * ctx->k + 1]);

    /* ---- ONNX Runtime inference ---- */
    int64_t   shape[]  = {1, state_dim};
    OrtValue *input    = NULL;
    OrtValue *output   = NULL;
    OrtStatus *status  = NULL;

    status = g_ort->CreateTensorWithDataAsOrtValue(
        ctx->mem_info, state, (size_t)state_dim * sizeof(float),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input);
    if (status) {
        fprintf(stderr, "[ddpg] ORT error: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        goto cleanup;
    }

    const char *input_names[]  = {"state"};
    const char *output_names[] = {"action"};

    status = g_ort->Run(ctx->session, NULL,
                        input_names, (const OrtValue *const *)&input, 1,
                        output_names, 1, &output);
    if (status) {
        fprintf(stderr, "[ddpg] ORT error: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        goto cleanup;
    }

    float *action_data = NULL;
    status = g_ort->GetTensorMutableData(output, (void **)&action_data);
    if (status) {
        fprintf(stderr, "[ddpg] ORT error: %s\n", g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        goto cleanup;
    }

    /* actor outputs tanh(·) ∈ [-1, 1]; treat as log2 cwnd scale factor */
    float action = action_data[0];
    float new_cwnd = ctx->current_cwnd * powf(2.0f, action);
    new_cwnd = fmaxf((float)MSS, fminf(1000.0f * (float)MSS, new_cwnd));

    DBG("action=%.4f old_cwnd=%.0f new_cwnd=%.0f\n",
        action, ctx->current_cwnd, new_cwnd);

    ctx->current_cwnd = new_cwnd;

cleanup:
    if (output) g_ort->ReleaseValue(output);
    if (input)  g_ort->ReleaseValue(input);
    free(state);
    free(entries);
}

static void ddpg_on_ack(void *raw_ctx, tcp_info_t *info) {
    ddpg_ctx_t *ctx = (ddpg_ctx_t *)raw_ctx;

    ctx->total_packets++;
    if (info->dup_ack || info->timeout)
        ctx->lost_packets++;

    queue_push(ctx->queue, info);

    if (ctx->inference_busy) {
        DBG("inference in progress, skipping\n");
        return;
    }
    if (queue_size(ctx->queue) < ctx->k) {
        DBG("queue not full yet (%d/%d), skipping inference\n",
            queue_size(ctx->queue), ctx->k);
        return;
    }

    ctx->inference_busy = 1;
    ddpg_run_inference(ctx);
    ctx->inference_busy = 0;
}

static void ddpg_on_timeout(void *raw_ctx, tcp_info_t *info) {
    ddpg_on_ack(raw_ctx, info);
}

static float ddpg_get_cwnd(void *raw_ctx) {
    return ((ddpg_ctx_t *)raw_ctx)->current_cwnd;
}

static void ddpg_destroy(void *raw_ctx) {
    ddpg_ctx_t *ctx = (ddpg_ctx_t *)raw_ctx;
    queue_destroy(ctx->queue);
    if (ctx->mem_info) g_ort->ReleaseMemoryInfo(ctx->mem_info);
    if (ctx->session)  g_ort->ReleaseSession(ctx->session);
    if (ctx->opts)     g_ort->ReleaseSessionOptions(ctx->opts);
    if (ctx->env)      g_ort->ReleaseEnv(ctx->env);
    free(ctx);
}

static cc_ops_t ddpg_ops = {
    .on_ack     = ddpg_on_ack,
    .on_timeout = ddpg_on_timeout,
    .get_cwnd   = ddpg_get_cwnd,
    .destroy    = ddpg_destroy,
};

cc_ops_t *ddpg_create(void **ctx_out, const char *model_dir) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "[ddpg] failed to get ONNX Runtime API\n");
        return NULL;
    }

    int k = read_k(model_dir);
    if (k <= 0) {
        fprintf(stderr, "[ddpg] invalid k=%d from %s\n", k, model_dir);
        return NULL;
    }

    ddpg_ctx_t *ctx   = calloc(1, sizeof(ddpg_ctx_t));
    ctx->k            = k;
    ctx->current_cwnd = (float)(INIT_CWND_SEGS * MSS);
    ctx->queue        = queue_create(k);

    OrtStatus *s;
    s = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ddpg_eval", &ctx->env);
    if (s) g_ort->ReleaseStatus(s);
    s = g_ort->CreateSessionOptions(&ctx->opts);
    if (s) g_ort->ReleaseStatus(s);

    char onnx_path[MAX_PATH];
    snprintf(onnx_path, sizeof(onnx_path), "%s/ddpg.onnx", model_dir);
    DBG("loading model from %s\n", onnx_path);

    OrtStatus *status = g_ort->CreateSession(ctx->env, onnx_path,
                                             ctx->opts, &ctx->session);
    if (status) {
        fprintf(stderr, "[ddpg] CreateSession failed: %s\n",
                g_ort->GetErrorMessage(status));
        g_ort->ReleaseStatus(status);
        queue_destroy(ctx->queue);
        g_ort->ReleaseSessionOptions(ctx->opts);
        g_ort->ReleaseEnv(ctx->env);
        free(ctx);
        return NULL;
    }

    s = g_ort->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0,
                                OrtMemTypeDefault, &ctx->mem_info);
    if (s) g_ort->ReleaseStatus(s);

    DBG("model loaded. k=%d state_dim=%d\n", k, k * 3 + 2);

    *ctx_out = ctx;
    return &ddpg_ops;
}
