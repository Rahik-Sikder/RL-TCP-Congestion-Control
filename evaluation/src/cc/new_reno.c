#include "new_reno.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef DEBUG
#define DBG(...) fprintf(stderr, "[new_reno] " __VA_ARGS__)
#else
#define DBG(...) ((void)0)
#endif

typedef struct {
    float cwnd;
    float ssthresh;
    int   dup_ack_count;
    float mss;
} new_reno_ctx_t;

static void nr_on_ack(void *raw_ctx, tcp_info_t *info) {
    new_reno_ctx_t *ctx = (new_reno_ctx_t *)raw_ctx;

    if (info->dup_ack) {
        ctx->dup_ack_count++;
        if (ctx->dup_ack_count == 3) {
            /* Fast retransmit */
            float new_ssthresh = ctx->cwnd / 2.0f;
            if (new_ssthresh < 2.0f * ctx->mss)
                new_ssthresh = 2.0f * ctx->mss;
            ctx->ssthresh      = new_ssthresh;
            ctx->cwnd          = ctx->ssthresh + 3.0f * ctx->mss;
            ctx->dup_ack_count = 0;
            DBG("fast retransmit triggered: ssthresh=%.0f cwnd=%.0f\n",
                ctx->ssthresh, ctx->cwnd);
        }
        return;
    }

    ctx->dup_ack_count = 0;

    if (ctx->cwnd < ctx->ssthresh) {
        /* Slow start */
        ctx->cwnd += ctx->mss;
    } else {
        /* Congestion avoidance (AIMD) */
        ctx->cwnd += ctx->mss * ctx->mss / ctx->cwnd;
    }
    DBG("on_ack: cwnd=%.0f ssthresh=%.0f\n", ctx->cwnd, ctx->ssthresh);
}

static void nr_on_timeout(void *raw_ctx, tcp_info_t *info) {
    new_reno_ctx_t *ctx = (new_reno_ctx_t *)raw_ctx;
    (void)info;

    float new_ssthresh = ctx->cwnd / 2.0f;
    if (new_ssthresh < 2.0f * ctx->mss)
        new_ssthresh = 2.0f * ctx->mss;
    ctx->ssthresh      = new_ssthresh;
    ctx->cwnd          = ctx->mss;   /* slow start restart */
    ctx->dup_ack_count = 0;
    DBG("timeout: cwnd reset to MSS, ssthresh=%.0f\n", ctx->ssthresh);
}

static float nr_get_cwnd(void *raw_ctx) {
    return ((new_reno_ctx_t *)raw_ctx)->cwnd;
}

static void nr_destroy(void *raw_ctx) {
    free(raw_ctx);
}

static cc_ops_t new_reno_ops = {
    .on_ack     = nr_on_ack,
    .on_timeout = nr_on_timeout,
    .get_cwnd   = nr_get_cwnd,
    .destroy    = nr_destroy,
};

cc_ops_t *new_reno_create(void **ctx_out) {
    new_reno_ctx_t *ctx = malloc(sizeof(new_reno_ctx_t));
    if (!ctx) return NULL;
    ctx->mss           = (float)MSS;
    ctx->cwnd          = (float)(INIT_CWND_SEGS * MSS);
    ctx->ssthresh      = 65536.0f;  /* 64 KB initial ssthresh */
    ctx->dup_ack_count = 0;
    DBG("created: cwnd=%.0f ssthresh=%.0f\n", ctx->cwnd, ctx->ssthresh);
    *ctx_out = ctx;
    return &new_reno_ops;
}
