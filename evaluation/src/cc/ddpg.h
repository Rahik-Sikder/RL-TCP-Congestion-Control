#ifndef DDPG_H
#define DDPG_H

#include "../transport/transport.h"
#include "cc.h"

/* DDPG constructor. model_dir must contain ddpg.onnx and ddpg.info.json */
cc_ops_t *ddpg_create(void **ctx, const char *model_dir);

#endif /* DDPG_H */
