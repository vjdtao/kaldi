SPEECHTOOLS=/opt/speech/tools
export KALDI_ROOT=$SPEECHTOOLS/kaldi/08092019

LM_ROOT=/opt/nlp/lmtools/srilm-1.7.2
export PATH=$KALDI_ROOT/egs/wsj/s5/utils/:$SPEECHTOOLS/openfst-1.6.7/bin:$LM_ROOT/bin:$LM_ROOT/bin/i686-m64:$PWD:$PATH

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib:/usr/local/lib64:/usr/local/cuda/bin/nvcc

export LC_ALL=C
