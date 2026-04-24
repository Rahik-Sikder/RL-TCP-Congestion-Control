#ifndef QUEUE_H
#define QUEUE_H

#include "transport.h"

typedef struct queue_t queue_t;

queue_t *queue_create(int capacity);
void     queue_push(queue_t *q, const tcp_info_t *info);
int      queue_read_all(queue_t *q, tcp_info_t *out_buf);
int      queue_size(const queue_t *q);
void     queue_destroy(queue_t *q);

#endif /* QUEUE_H */
