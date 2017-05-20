#!/bin/bash

DATA_DIR="../data"
TWTR_DIR="../twitter-datasets"

function fail {
  echo >&2 $1
  exit 1
}

platform='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
elif [[ "$unamestr" == 'FreeBSD' ]]; then
   platform='freebsd'
fi

if ! [[ -d "${TWTR_DIR}" ]]; then
  fail "Twitter datasets not downloaded"
fi

if ! [[ -d "${DATA_DIR}" ]]; then
  mkdir -p ${DATA_DIR}
  echo "Data directory created"
fi

if ! [[ -f "${DATA_DIR}/full_train.txt" ]]; then
  echo "Making organized dataset"
  python3 organize_dataset.py ../twitter-datasets ../data || fail "Could not create organized dataset"
fi
echo "organized dataset ready"

if ! [[ -f "${DATA_DIR}/vocab.txt" ]]; then
  echo "Making vocabulary"
  if [[ ${platform} == "linux" ]]; then
      cat ../twitter-datasets/train_pos.txt ../twitter-datasets/train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > ../data/vocab.txt || fail "Could not create vocab"
  else
      cat ../twitter-datasets/train_pos.txt ../twitter-datasets/train_neg.txt |
      sed -e 's/ /\'$'\n/g' | grep -v "^\s*$" | sort | uniq -c > ../data/vocab.txt || fail "Could not create vocab"
  fi
fi
echo "vocabulary ready"

if ! [[ -f "${DATA_DIR}/vocab_trimmed.txt" ]]; then
  echo "Trimming Vocabulary"
  if [[ ${platform} == "linux" ]]; then
    cat ../data/vocab.txt | sort -n -r | head -n 20000 > ../data/vocab_trimmed.txt || fail "could not trim vocabulary"
  else
    cat ../data/vocab.txt | sort -n -r | head -n 20000 > ../data/vocab_trimmed.txt || fail "could not trim vocabulary"
#   cat ../data/vocab.txt | sed "s/^\s\+//g" | sed "s/\s\+/ /g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > ../data/vocab_cut.txt || fail "could not cut vocab"
  fi
fi
echo "trimmed vocabulary ready"

if ! [[ -f "${DATA_DIR}/vocab.pkl" ]]; then
  echo "Dumping data"
  python3 pickle_vocab.py ${DATA_DIR} || fail "could not create pickle"
fi

echo "Make encodings of datasets"
if ! [[ -f "${DATA_DIR}/tf_encoded_small_train.pkl" ]]; then
  echo "Encoding the small dataset"
  python3 tf_encoder.py  || fail "could not create encoder"
fi
