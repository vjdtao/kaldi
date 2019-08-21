#!/bin/bash

#	2019 Jake Tao
# This recipe does multi-style training of TDNN model

# local/chain/compare_wer_general.sh --rt03 tdnn7q_sp tdnn1a_aug
# System                tdnn7q_sp tdnn1a_aug
# WER on train_dev(tg)      11.91     12.06
# Final train prob         -0.062    -0.087
# Final valid prob         -0.074    -0.105
# Final train prob (xent)        -0.933    -1.164
# Final valid prob (xent)       -0.9027   -1.2246
# Num-parameters               18693376  18483664

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
num_epochs=3

# Augmentation options
aug_list="reverb babble music noise clean" # Original train dir is referred to as `clean`
num_reverb_copies=1
use_ivectors=true

affix=1a
suffix="_aug"

decode_iter=
decode_nj=10
test_sets="eval1 eval2 eval3"

# training options
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
num_jobs_initial=3
num_jobs_final=10
minibatch_size=128,64
frames_per_eg=150,140,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=exp/chain/tdnn${affix}${suffix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

clean_set=train_nodup
clean_ali=tri4_ali_nodup
train_set=$clean_set$suffix # Will be prepared by the script local/nnet3/prepare_multistyle_data.sh
ali_dir=$clean_ali$suffix
treedir=exp/chain/tree$suffix
lang=data/lang_chain$suffix


# First creates augmented data and then extracts features for it data
# The script also creates alignments for aug data by copying clean alignments
local/nnet3/multi_condition/run_aug_common.sh --stage $stage \
  --aug-list "$aug_list" --num-reverb-copies $num_reverb_copies \
  --use-ivectors "$use_ivectors" \
  --train-set $clean_set --clean-ali $clean_ali || exit 1;

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  prefixes=""
  include_original=false
  for n in $aug_list; do
    if [ "$n" == "reverb" ]; then
      for i in `seq 1 $num_reverb_copies`; do
        prefixes="$prefixes "reverb$i
      done
    elif [ "$n" != "clean" ]; then
      prefixes="$prefixes "$n
    else
      # The original train directory will not have any prefix
      # include_original flag will take care of copying the original lattices
      include_original=true
    fi
  done
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${clean_set} \
    data/lang exp/tri4 exp/tri4_lats_nodup${suffix}_clean
  rm exp/tri4_lats_nodup${suffix}_clean/fsts.*.gz # save space
  steps/copy_lat_dir.sh --nj $nj --cmd "$train_cmd" \
    --include-original "$include_original" --prefixes "$prefixes" \
    data/${train_set} exp/tri4_lats_nodup${suffix}_clean exp/tri4_lats_nodup${suffix} || exit 1;
fi

if [ $stage -le 12 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 13 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang exp/$ali_dir $treedir
fi

if [ $stage -le 14 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=625
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn3 dim=625
  relu-batchnorm-layer name=tdnn4 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn5 dim=625
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=625

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn9 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn9 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
    --dir $dir  || exit 1;

fi

if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_csj_tg $dir $dir/graph_csj_tg

  for decode_set in $test_sets; do
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set}$suffix \
          $dir/graph_csj_tg data/${decode_set}_hires $dir/decode_${decode_set}
  done

  steps/online/nnet3/prepare_online_decoding.sh \
      --mfcc-config conf/mfcc_hires.conf \
      $lang exp/nnet3/extractor $dir ${dir}_online

  for decode_set in $test_sets; do
      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $dir/graph_csj_tg data/${decode_set}_hires \
         ${dir}_online/decode_${decode_set}
  done
fi

exit 0;
