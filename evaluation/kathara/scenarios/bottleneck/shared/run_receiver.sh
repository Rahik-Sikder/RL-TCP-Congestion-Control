#!/bin/sh
export LD_LIBRARY_PATH=/shared/lib
exec /shared/receiver "$@"
