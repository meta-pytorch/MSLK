#/bin/bash

# This script is to be run after important a new version of
# https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute
# in order to fix all the import references and have them point to MSLK
# rather than the original flash_attn repo

for f in *.py; do
    sed -i '1s/^# @nolint # fbcode$/&/;t;1s/^/# @nolint # fbcode\n/' $f
    sed -i 's/from flash_attn\.cute/from mslk.attention.flash_attn/g' $f
    sed -i 's/import flash_attn\.cute/import mslk.attention.flash_attn/g' $f
done
