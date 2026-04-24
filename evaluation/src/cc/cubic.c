#include "cubic.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef DEBUG
#define DBG(...) fprintf(stderr, "[cubic] " __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

#define CUBIC_C    0.4f
#define CUBIC_BETA 0.7f

typedef struct {
    float    cwnd;
    float    W_max;
    float    K;
    uint64_t t_epoch_us;
    float    mss;
} cubic_ctx_t;

static void cubic_on_ack(void *raw_ctx, tcp_info_t *info) {
    cubic_ctx_t *ctx = (cubic_ctx_t *)raw_ctx;
    (void)info;

    float t = (float)(now_us() - ctx->t_epoch_us) / 1e6f;
    float diff = t - ctx->K;
    float W_cubic = CUBIC_C * diff * diff * diff + ctx->W_max;

    if (W_cubic > ctx->cwnd) {
        ctx->cwnd = W_cubic;
        if (ctx->cwnd > 1000.0f * ctx->mss)
            ctx->cwnd = 1000.0f * ctx->mss;
        if (ctx->cwnd < ctx->mss)
            ctx->cwnd = ctx->mss;
    }
    DBG("on_ack: t=%.3fs K=%.3f W_cubic=%.0f cwnd=%.0f\n",
        t, ctx->K, W_cubic, ctx->cwnd);
}

static void cubic_on_timeout(void *raw_ctx, tcp_info_t *info) {
    cubic_ctx_t *ctx = (cubic_ctx_t *)raw_ctx;
    (void)info;

    ctx->W_max      = ctx->cwnd;
    ctx->cwnd       = ctx->cwnd * CUBIC_BETA;
    if (ctx->cwnd < ctx->mss) ctx->cwnd = ctx->mss;
    ctx->K          = cbrtf(ctx->W_max * (1.0f - CUBIC_BETA) / CUBIC_C);
    ctx->t_epoch_us = now_us();
    DBG("timeout: W_max=%.0f K=%.3f new_cwnd=%.0f\n",
        ctx->W_max, ctx->K, ctx->cwnd);
}

static float cubic_get_cwnd(void *raw_ctx) {
    return ((cubic_ctx_t *)raw_ctx)->cwnd;
}

static void cubic_destroy(void *raw_ctx) {
    free(raw_ctx);
}

static cc_ops_t cubic_ops = {
    .on_ack     = cubic_on_ack,
    .on_timeout = cubic_on_timeout,
    .get_cwnd   = cubic_get_cwnd,
    .destroy    = cubic_destroy,
};

cc_ops_t *cubic_create(void **ctx_out) {
    cubic_ctx_t *ctx = malloc(sizeof(cubic_ctx_t));
    if (!ctx) return NULL;
    ctx->mss        = (float)MSS;
    ctx->cwnd       = (float)(INIT_CWND_SEGS * MSS);
    ctx->W_max      = ctx->cwnd;
    ctx->K          = 0.0f;
    ctx->t_epoch_us = now_us();
    DBG("created: cwnd=%.0f W_max=%.0f\n", ctx->cwnd, ctx->W_max);
    *ctx_out = ctx;
    return &cubic_ops;
}
