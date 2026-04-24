#ifndef TRANSPORT_H
#define TRANSPORT_H

#include <stdint.h>
#include <stdio.h>

#define MSS           1400      /* max segment size, bytes */
#define MAX_WINDOW    1024      /* max in-flight packets tracked */
#define TIMEOUT_US    200000    /* 200ms timeout in microseconds */
#define INIT_CWND_SEGS 10      /* initial cwnd in segments */

/* Per-packet telemetry — our userspace equivalent of kernel tcp_info */
typedef struct {
    float rtt_ms;    /* measured RTT for this packet */
    int   dup_ack;   /* 1 = this ACK was a duplicate */
    int   timeout;   /* 1 = this packet timed out */
    float cwnd;      /* cwnd at send time, in bytes */
} tcp_info_t;

/* Pluggable congestion-control interface. All three algorithms implement this. */
typedef struct {
    void  (*on_ack)(void *ctx, tcp_info_t *info);
    void  (*on_timeout)(void *ctx, tcp_info_t *info);
    float (*get_cwnd)(void *ctx);   /* returns current cwnd in bytes */
    void  (*destroy)(void *ctx);
} cc_ops_t;

/* Wire format: packet header (sender → receiver) */
typedef struct __attribute__((packed)) {
    uint32_t seq;
    uint16_t data_len;
    uint64_t send_time_us;
} pkt_hdr_t;

/* Wire format: ACK (receiver → sender) */
typedef struct __attribute__((packed)) {
    uint32_t seq;
} ack_hdr_t;

/* Per-in-flight-packet tracking */
typedef struct {
    uint64_t send_time_us;
    uint32_t seq;    /* sequence number of the packet occupying this slot */
    int      acked;
} inflight_t;

/* Transport instance */
typedef struct {
    int        sock_fd;
    cc_ops_t  *cc;
    void      *cc_ctx;

    /* sequence tracking */
    uint32_t   next_seq;
    uint32_t   base_seq;
    inflight_t window[MAX_WINDOW];

    /* dupACK tracking */
    uint32_t   last_acked_seq;
    int        dup_ack_count;

    /* stats */
    uint64_t   total_sent;
    uint64_t   total_acked;
    uint64_t   total_timeouts;
    double     rtt_sum_ms;
    uint64_t   rtt_count;
    uint64_t   start_time_us;

    /* output */
    FILE      *ts_file;   /* timeseries.csv, owned by caller */
} transport_t;

/* Returns microseconds since epoch (CLOCK_MONOTONIC) */
uint64_t now_us(void);

transport_t *transport_create(int sock_fd, cc_ops_t *cc, void *cc_ctx, FILE *ts_file);
void         transport_run(transport_t *t, size_t total_bytes);
void         transport_destroy(transport_t *t);

#endif /* TRANSPORT_H */
