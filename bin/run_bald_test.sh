# DATASET: Bald
# Layers: 39 25
# Batch Size: 5
# Percentages to test: 0.01, 0.02, 0.03, 0.04, 0.05

# Linear (linear)
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.01 --printout ~/results/linear/bald-1-10.csv
#
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.02 --printout ~/results/linear/bald-2-10.csv
#
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.03 --printout ~/results/linear/bald-3-10.csv
#
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.04 --printout ~/results/linear/bald-4-10.csv
#
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat linear --batch-size 10000 --neg-cont 0.05 --printout ~/results/linear/bald-5-10.csv
#
## Probabilistic (prob)
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.01 --printout ~/results/prob/bald-1-10.csv
#
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.02 --printout ~/results/prob/bald-2-10.csv
#
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.03 --printout ~/results/prob/bald-3-10.csv
#
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.04 --printout ~/results/prob/bald-4-10.csv
#
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-1.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-2.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-3.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-4.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-5.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-6.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-7.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-8.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-9.csv && \
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat prob --batch-size 10000 --neg-cont 0.05 --printout ~/results/prob/bald-5-10.csv
#
## Neural Network (nn)
#python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.01 --printout ~/results/nn/bald-1-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.02 --printout ~/results/nn/bald-2-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.03 --printout ~/results/nn/bald-3-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.04 --printout ~/results/nn/bald-4-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat nn --batch-size 10000 --neg-cont 0.05 --printout ~/results/nn/bald-5-10.csv

# Ensemble (ensemble)
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.01 --printout ~/results/ensemble/bald-1-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.02 --printout ~/results/ensemble/bald-2-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.03 --printout ~/results/ensemble/bald-3-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.04 --printout ~/results/ensemble/bald-4-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat ensemble --batch-size 10000 --neg-cont 0.05 --printout ~/results/ensemble/bald-5-10.csv

# Proximity (proximity)
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.01 --printout ~/results/proximity/bald-1-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.02 --printout ~/results/proximity/bald-2-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.03 --printout ~/results/proximity/bald-3-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.04 --printout ~/results/proximity/bald-4-10.csv

python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-1.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-2.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-3.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-4.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-5.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-6.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-7.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-8.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-9.csv && \
python -m pyntrainer --mode eval --input-file https://happy-research.s3-ap-southeast-1.amazonaws.com/bald.csv --layers 39 25 --eval-cat proximity --batch-size 10000 --neg-cont 0.05 --printout ~/results/proximity/bald-5-10.csv
