#ifndef DQN_H
#define DQN_H

#include "../transport/transport.h"
#include "cc.h"

/* DQN constructor. model_dir must contain dqn.onnx and dqn.info.json */
cc_ops_t *dqn_create(void **ctx, const char *model_dir);

#endif /* DQN_H */
