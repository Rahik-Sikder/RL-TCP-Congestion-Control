#ifndef PPO_H
#define PPO_H

#include "../transport/transport.h"
#include "cc.h"

/* PPO constructor. model_dir is the directory containing ppo.onnx and ppo.info.json */
cc_ops_t *ppo_create(void **ctx, const char *model_dir);

#endif /* PPO_H */
