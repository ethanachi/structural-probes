#!/bin/bash

langs=( "ar" "de" "en" "es" "fa" "fi" "fr" "id" "zh" );

rm -rf /u/scr/ethanchi/all
mkdir /u/scr/ethanchi/all
for lang in ${langs[*]}; do
  for f in /u/scr/ethanchi/$lang/*{.conllu,*.txt}; do
    echo $f
    base=$(basename $f)
    echo $base
    cat $f >> /u/scr/ethanchi/all/$base;  
  done
done

echo "Loop finished."

for holdoutLang in ${langs[*]}; do
  rm -rf "/u/scr/ethanchi/holdout_$holdoutLang";
  mkdir /u/scr/ethanchi/holdout_$holdoutLang
  echo "Making holdout language $holdoutLang"
  for lang in ${langs[*]}; do
    echo "Comparison:" $lang $holdoutLang
    if [[ $lang == $holdoutLang ]]; then
      echo "Skipping $lang"; 
    else
      echo "Not skipping $lang"
      for f in /u/scr/ethanchi/$lang/*{.conllu,*.txt}; do
        echo $f
        base=$(basename $f)
        # echo $base;
        cat $f >> /u/scr/ethanchi/holdout_$holdoutLang/$base
      done
    fi
  done
done

langs=(de en es fr);

rm -rf /u/scr/ethanchi/all4
mkdir /u/scr/ethanchi/all4
for lang in ${langs[*]}; do
  for f in /u/scr/ethanchi/$lang/*{.conllu,*.txt}; do
    echo $f
    base=$(basename $f)
    echo $base
    cat $f >> /u/scr/ethanchi/all4/$base;  
  done
done

echo "Loop finished."

for holdoutLang in ${langs[*]}; do
  rm -rf "/u/scr/ethanchi/holdout4_$holdoutLang";
  mkdir /u/scr/ethanchi/holdout4_$holdoutLang
  echo "Making holdout language $holdoutLang"
  for lang in ${langs[*]}; do
    echo $lang
    if [[ $lang == $holdoutLang ]]; then
      echo "Skipping $lang"; 
    else
      echo "Not skipping $lang"
      for f in /u/scr/ethanchi/$lang/*{.conllu,*.txt}; do
        echo $f
        base=$(basename $f)
        # echo $base;
        cat $f >> /u/scr/ethanchi/holdout4_$holdoutLang/$base
      done
    fi
  done
done
