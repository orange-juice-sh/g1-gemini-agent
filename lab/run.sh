#!/bin/bash
# lab サービスの中で学習・再生する: bash /project/lab/run.sh train|play [引数]
cd /project/lab || exit 1
mkdir -p logs
/workspace/isaaclab/isaaclab.sh -p /project/lab/train.py "$@" --headless
status=$?
# コンテナは root で動くので、作ったファイルを自分の UID に戻す
if [ -d /project/lab/logs ]; then chown -R "${LOCAL_UID:-1000}:${LOCAL_GID:-1000}" /project/lab/logs; fi
exit $status
