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
  mkdir -p DATA_DIR
  echo "Data directory created"
fi

if ! [[ -f "${DATA_DIR}/full_train.txt" ]]; then
  echo "Making organized dataset"
  python3 organize_dataset.py ../twitter-datasets ../data || fail "Could not create orgranized dataset"
fi
echo "organized dataset ready"

if ! [[ -f "${DATA_DIR}/vocab.txt" ]]; then
  echo "Making vocabulary"
  if [[ ${platform} == "linux" ]]; then
      cat ../twitter-datasets/train_pos.txt ../twitter-datasets/train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c | sort -d -n -f1 -r | head -n 20000 > ../data/vocab.txt || fail "Could not create vocab"
  else
      cat ../twitter-datasets/train_pos.txt ../twitter-datasets/train_neg.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c | sort -d -n -k1 -r | head -n 20000 > ../data/vocab.txt || fail "Could not create vocab"
  fi
fi
echo "vocabulary ready"

if ! [[ -f "${DATA_DIR}/vocab_cut.txt" ]]; then
  echo "Trimming Vocabulary"
  cat ../data/vocab.txt | sed "s/^\s\+//g" | sed "s/\s\+/ /g" | sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > ../data/vocab_cut.txt || fail "could not cut vocab"
fi
echo "trimmed vocabulary ready"

if ! [[ -f "${DATA_DIR}/vocab.pkl" ]]; then
  echo "Dumping data"
  python3 pickle_vocab.py ${DATA_DIR} || fail "could not create pickle"
fi

echo "Make encodings of datasets"
