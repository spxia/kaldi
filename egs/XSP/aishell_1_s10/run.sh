#!/bin/bash

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
#           2020 Xiaomi Corporation (author: Haowen Qiu)
# Apache 2.0

# This file demonstrates how to run LF-MMI training in PyTorch
# with kaldi pybind. The neural network part is based on PyTorch,
# while everything else is based on kaldi.
#
# We assume that you have built kaldi pybind and installed PyTorch.
# You also need a GPU to run this example.
#
# PyTorch with version `1.3.0dev20191006` has been tested and is
# known to work.
#
# Note that we have used Tensorboard to visualize the training loss.
# You do **NOT** need to install TensorFlow to use Tensorboard.

. ./cmd.sh
. ./path.sh

data=/home/asr-corpus/public/aishell
data_url=www.openslr.org/resources/33

nj=30

stage=13
chain_stage=21
. utils/parse_options.sh || exit 1;

if [[ $stage -le 0 ]]; then
  local/download_and_untar.sh $data $data_url data_aishell || exit 1
  local/download_and_untar.sh $data $data_url resource_aishell || exit 1
fi

if [[ $stage -le 1 ]]; then
  local/aishell_prepare_dict.sh $data/resource_aishell || exit 1
  # generated in data/local/dict
fi

if [[ $stage -le 2 ]]; then
  local/aishell_data_prep.sh $data/data_aishell/wav \
    $data/data_aishell/transcript || exit 1
  # generated in data/{train,test,dev}/{spk2utt text utt2spk wav.scp}
fi

if [[ $stage -le 3 ]]; then
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
      "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1
fi

if [[ $stage -le 4 ]]; then
  local/aishell_train_lms.sh || exit 1
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1
  cp data/lang/phones/* data/lang_test/phones/
fi

if [[ $stage -le 5 ]]; then
  for x in train dev test; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$x  || exit 1
    steps/compute_cmvn_stats.sh data/$x || exit 1
    utils/fix_data_dir.sh data/$x || exit 1
  done
fi

if [[ $stage -le 6 ]]; then
  steps/train_mono.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/mono || exit 1
fi

if [[ $stage -le 7 ]]; then
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/mono exp/mono_ali || exit 1
fi

if [[ $stage -le 8 ]]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
   2500 20000 data/train data/lang exp/mono_ali exp/tri1 || exit 1
fi

if [[ $stage -le 9 ]]; then
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1
fi

if [[ $stage -le 10 ]]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
   3000 40000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1
fi

if [[ $stage -le 11 ]]; then
  steps/align_si.sh --cmd "$train_cmd" --nj $nj \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1
fi

if [[ $stage -le 12 ]]; then
  steps/train_sat.sh --cmd "$train_cmd" \
   4000 80000 data/train data/lang exp/tri2_ali exp/tri3 || exit 1
fi

if [[ $stage -le 13 ]]; then
  local/run_chain.sh --nj $nj --stage $chain_stage
fi
