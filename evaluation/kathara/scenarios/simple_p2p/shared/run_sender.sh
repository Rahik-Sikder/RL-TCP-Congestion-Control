#!/bin/sh
export LD_LIBRARY_PATH=/shared/lib
mkdir -p /tmp/results
exec /shared/sender "$@"
