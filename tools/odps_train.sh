#!/usr/bin/env bash

set -x

grep -r "ipdb" \
    --exclude-dir .git \
    --exclude-dir data \
    --exclude-dir pretrained \
    --exclude-dir work_dirs \
    --exclude ./tools/odps_train.sh \
    --exclude ./tools/odps_test.sh \
    --exclude requirements.txt \
    .
if [[ $? -eq 0 ]]; then
    echo "ipdb is not allowed in this repo"
    exit 1
fi

set -e

JOB_NAME=$1
CONFIG=$2
GPUS=$3
PY_ARGS=${@:4}

PROJECT_NAME=${PROJECT_NAME:-HEAD}
ENTRY_FILE=${ENTRY_FILE:-tools/train.py}
WORKBENCH=${WORKBENCH:-search_algo_quality_dev}  # search_algo_quality_dev, imac_dev
ROLEARN=${ROLEARN:-searchalgo}  # searchalgo, imac

tar -zchf /tmp/${PROJECT_NAME}.tar.gz \
    --exclude .git \
    --exclude data \
    --exclude pretrained \
    --exclude work_dirs \
    .

cmd_oss="
use ${WORKBENCH};
pai -name pytorch180
    -Dscript=\"file:///tmp/${PROJECT_NAME}.tar.gz\"
    -DentryFile=\"${ENTRY_FILE}\"
    -DworkerCount=${GPUS}
    -DuserDefinedParameters=\"${CONFIG} --cfg-options checkpoint_config.create_symlink:False evaluation.tmpdir:\\\'work_dirs/tmp${RANDOM}\\\' --odps --work-dir work_dirs/${JOB_NAME} --launcher pytorch ${PY_ARGS}\"
    -Dbuckets=\"oss://mvap-data/zhax/wangluting/?role_arn=acs:ram::1367265699002728:role/${ROLEARN}4pai&host=cn-zhangjiakou.oss.aliyuncs.com\";
"
# set odps.algo.hybrid.deploy.info=LABEL:V100:OPER_EQUAL;
    # -Dcluster=\"{\\\"worker\\\":{\\\"gpu\\\":${GPUS}00}}\"
odpscmd -e "${cmd_oss}"
