#ifndef CC_H
#define CC_H

#include "../transport/transport.h"

/*
 * Constructor type: each CC module exposes a function matching this signature.
 * Returns the cc_ops_t table and sets *ctx to the algorithm's private state.
 * Caller must call ops->destroy(ctx) when done.
 */
typedef cc_ops_t *(*cc_constructor_t)(void **ctx);

/* NewReno constructor — defined in cc/new_reno.c */
cc_ops_t *new_reno_create(void **ctx);

/* Cubic constructor — defined in cc/cubic.c */
cc_ops_t *cubic_create(void **ctx);

#endif /* CC_H */
