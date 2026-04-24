/*
 * transport.c — UDP-based reliable sender event loop
 * Implements the transport API declared in transport.h (C11).
 */

#include "transport.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <time.h>

/* -------------------------------------------------------------------------
 * Internal helpers (forward declarations)
 * ---------------------------------------------------------------------- */
static void drain_acks(transport_t *t);
static void check_timeouts(transport_t *t);
static int  count_inflight(const transport_t *t);

/* =========================================================================
 * now_us — microseconds since CLOCK_MONOTONIC epoch
 * ====================================================================== */
uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)(ts.tv_nsec / 1000);
}

/* =========================================================================
 * transport_create
 * ====================================================================== */
transport_t *transport_create(int sock_fd, cc_ops_t *cc, void *cc_ctx,
                              FILE *ts_file)
{
    transport_t *t = calloc(1, sizeof(transport_t));
    if (!t) return NULL;

    t->sock_fd      = sock_fd;
    t->cc           = cc;
    t->cc_ctx       = cc_ctx;
    t->ts_file      = ts_file;
    t->start_time_us = now_us();

    /* Mark every window slot as acked/empty */
    for (int i = 0; i < MAX_WINDOW; i++) {
        t->window[i].acked = 1;
        t->window[i].send_time_us = 0;
    }

    float initial_cwnd = cc->get_cwnd(cc_ctx);

#ifdef DEBUG
    fprintf(stderr, "[transport] created: sock_fd=%d initial_cwnd=%.0f bytes\n",
            sock_fd, (double)initial_cwnd);
#else
    (void)initial_cwnd;
#endif

    if (ts_file) {
        fprintf(ts_file,
                "timestamp_ms,cwnd_bytes,rtt_ms,dup_ack_count,timeout_count\n");
        fflush(ts_file);
    }

    return t;
}

/* =========================================================================
 * transport_run — main send/receive event loop
 * ====================================================================== */
void transport_run(transport_t *t, size_t total_bytes)
{
    static uint8_t payload[MSS];
    static int payload_initialized = 0;
    if (!payload_initialized) {
        memset(payload, 0xAB, MSS);
        payload_initialized = 1;
    }

    size_t bytes_sent = 0;   /* application bytes handed to the socket */

    while (bytes_sent < total_bytes || count_inflight(t) > 0) {

        drain_acks(t);
        check_timeouts(t);

        float cwnd_bytes  = t->cc->get_cwnd(t->cc_ctx);
        int   max_inflight = (int)(cwnd_bytes / (float)MSS);
        if (max_inflight < 1) max_inflight = 1;

        /* Send as many new packets as the window allows */
        while (count_inflight(t) < max_inflight && bytes_sent < total_bytes) {

            uint32_t seq = t->next_seq++;
            int      idx = (int)(seq % MAX_WINDOW);

            /* Determine chunk size for this packet */
            size_t remaining = total_bytes - bytes_sent;
            uint16_t data_len = (remaining >= MSS) ? MSS : (uint16_t)remaining;

            /* Build packet header */
            pkt_hdr_t hdr;
            hdr.seq          = seq;
            hdr.data_len     = data_len;
            hdr.send_time_us = now_us();

            /* sendmsg: header iovec + payload iovec */
            struct iovec iov[2];
            iov[0].iov_base = &hdr;
            iov[0].iov_len  = sizeof(hdr);
            iov[1].iov_base = payload;
            iov[1].iov_len  = data_len;

            struct msghdr msg;
            memset(&msg, 0, sizeof(msg));
            msg.msg_iov    = iov;
            msg.msg_iovlen = 2;

            ssize_t sent = sendmsg(t->sock_fd, &msg, 0);
            if (sent < 0) {
                /* Non-fatal: skip this slot and try later */
                t->next_seq--;
                break;
            }

            /* Record in window */
            t->window[idx].send_time_us = hdr.send_time_us;
            t->window[idx].seq          = seq;
            t->window[idx].acked        = 0;

            bytes_sent    += data_len;
            t->total_sent++;             /* count packets, not bytes */

#ifdef DEBUG
            fprintf(stderr,
                    "[transport] SEND seq=%u chunk=%u bytes_sent=%zu\n",
                    seq, (unsigned)data_len, bytes_sent);
#endif
        }
    }
}

/* =========================================================================
 * drain_acks — non-blocking ACK drain (static)
 * ====================================================================== */
static void drain_acks(transport_t *t)
{
    int drained = 0;

    for (;;) {
        /* 1 ms timeout on select so the main loop stays responsive */
        fd_set rset;
        FD_ZERO(&rset);
        FD_SET(t->sock_fd, &rset);
        struct timeval tv = { .tv_sec = 0, .tv_usec = 1000 };

        int ready = select(t->sock_fd + 1, &rset, NULL, NULL, &tv);
        if (ready <= 0) break;   /* timeout or error — nothing more to read */

        ack_hdr_t ack;
        ssize_t   n = recv(t->sock_fd, &ack, sizeof(ack), 0);
        if (n != (ssize_t)sizeof(ack)) break;

        drained++;

        uint32_t seq = ack.seq;
        int      idx = (int)(seq % MAX_WINDOW);

        /* Ignore ACKs for already-retired slots */
        if (t->window[idx].acked || t->window[idx].send_time_us == 0 ||
            t->window[idx].seq != ack.seq) {
#ifdef DEBUG
            fprintf(stderr, "[transport] ACK seq=%u ignored (slot mismatch or already acked)\n", ack.seq);
#endif
            continue;
        }

        uint64_t now       = now_us();
        uint64_t age_us    = now - t->window[idx].send_time_us;
        float    rtt_ms    = (float)age_us / 1000.0f;

        /* dupACK detection */
        int is_dup = 0;
        if (seq == t->last_acked_seq) {
            t->dup_ack_count++;
            is_dup = 1;
        } else {
            t->dup_ack_count  = 0;
            t->last_acked_seq = seq;
        }

        /* Mark slot retired */
        t->window[idx].acked = 1;
        t->total_acked++;
        t->rtt_sum_ms += rtt_ms;
        t->rtt_count++;

        /* Notify CC */
        float cwnd_now = t->cc->get_cwnd(t->cc_ctx);
        tcp_info_t info = {
            .rtt_ms  = rtt_ms,
            .dup_ack = is_dup,
            .timeout = 0,
            .cwnd    = cwnd_now,
        };
        t->cc->on_ack(t->cc_ctx, &info);

        /* Timeseries CSV row */
        if (t->ts_file) {
            float new_cwnd   = t->cc->get_cwnd(t->cc_ctx);
            double elapsed_ms =
                (double)(now - t->start_time_us) / 1000.0;
            fprintf(t->ts_file, "%.3f,%.0f,%.3f,%d,%llu\n",
                    elapsed_ms,
                    (double)new_cwnd,
                    (double)rtt_ms,
                    t->dup_ack_count,
                    (unsigned long long)t->total_timeouts);
            fflush(t->ts_file);
        }

#ifdef DEBUG
        fprintf(stderr,
                "[transport] ACK seq=%u rtt_ms=%.3f cwnd=%.0f dup=%d\n",
                seq, (double)rtt_ms,
                (double)t->cc->get_cwnd(t->cc_ctx),
                is_dup);
#endif
    }

#ifdef DEBUG
    fprintf(stderr, "[transport] drain_acks: drained %d ACK(s)\n", drained);
#else
    (void)drained;
#endif
}

/* =========================================================================
 * check_timeouts — retire timed-out in-flight packets (static)
 * ====================================================================== */
static void check_timeouts(transport_t *t)
{
    uint64_t now = now_us();

    for (int i = 0; i < MAX_WINDOW; i++) {
        if (t->window[i].acked || t->window[i].send_time_us == 0)
            continue;

        uint64_t age_us = now - t->window[i].send_time_us;
        if (age_us <= TIMEOUT_US)
            continue;

        /* Retire the slot */
        t->window[i].acked = 1;
        t->total_timeouts++;

        float cwnd_now = t->cc->get_cwnd(t->cc_ctx);
        tcp_info_t info = {
            .rtt_ms  = (float)age_us / 1000.0f,
            .dup_ack = 0,
            .timeout = 1,
            .cwnd    = cwnd_now,
        };
        t->cc->on_timeout(t->cc_ctx, &info);

#ifdef DEBUG
        /* Recover the sequence number from window index (best effort) */
        fprintf(stderr,
                "[transport] TIMEOUT slot=%d age_us=%llu\n",
                i, (unsigned long long)age_us);
#endif
    }
}

/* =========================================================================
 * count_inflight — count active (un-acked) window slots (static)
 * ====================================================================== */
static int count_inflight(const transport_t *t)
{
    int n = 0;
    for (int i = 0; i < MAX_WINDOW; i++) {
        if (!t->window[i].acked && t->window[i].send_time_us != 0)
            n++;
    }
    return n;
}

/* =========================================================================
 * transport_destroy
 * ====================================================================== */
void transport_destroy(transport_t *t)
{
    free(t);
}
