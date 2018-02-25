#!/bin/bash
bash 
declare -a langs=("hiligaynon", "middle-english", "plautdietsch", "xhosa")

for lang in "${langs[@]}"
do
	echo $lang'.txt.'
	exec java -jar ./negsel2.jar -self english.train -n 10 -r 4 -c -l < 'lang/'$lang'.txt' > '../results/'$lang'.results'
done
