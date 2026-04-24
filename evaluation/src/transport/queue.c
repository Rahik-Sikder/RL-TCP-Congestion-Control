#include "queue.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct queue_t {
    tcp_info_t *buf;
    int         cap;
    int         head;
    int         tail;
    int         count;
};

queue_t *queue_create(int capacity) {
    queue_t *q = malloc(sizeof(queue_t));
    if (!q) return NULL;
    q->buf = malloc((size_t)capacity * sizeof(tcp_info_t));
    if (!q->buf) { free(q); return NULL; }
    q->cap   = capacity;
    q->head  = 0;
    q->tail  = 0;
    q->count = 0;
    return q;
}

// TODO: Sampling improvement
// Instead of dropping head entries when full, split the ring buffer into k equal
// buckets. Each incoming packet replaces the oldest entry in its corresponding
// bucket (based on arrival index mod k). When inference runs, take one sample
// from each bucket in order. This provides uniform coverage across the full
// window even under high packet arrival rates, rather than biasing toward the
// most recent k packets.
void queue_push(queue_t *q, const tcp_info_t *info) {
    if (q->count == q->cap) {
        /* Drop head (oldest entry) to make room */
        q->head = (q->head + 1) % q->cap;
        q->count--;
#ifdef DEBUG
        fprintf(stderr, "[queue] DROP HEAD size=%d\n", q->count);
#endif
    }
    q->buf[q->tail] = *info;
    q->tail  = (q->tail + 1) % q->cap;
    q->count++;
#ifdef DEBUG
    fprintf(stderr, "[queue] PUSH size=%d/%d\n", q->count, q->cap);
#endif
}

int queue_read_all(queue_t *q, tcp_info_t *out_buf) {
    int idx = q->head;
    for (int i = 0; i < q->count; i++) {
        out_buf[i] = q->buf[idx];
        idx = (idx + 1) % q->cap;
    }
    return q->count;
}

int queue_size(const queue_t *q) {
    return q->count;
}

void queue_destroy(queue_t *q) {
    free(q->buf);
    free(q);
}
