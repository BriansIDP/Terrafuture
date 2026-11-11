import json
import sys, os


infile = sys.argv[1]
with open(infile) as fin:
    data = json.load(fin)
score = 0
total = 0
for datapiece in data:
    score += datapiece["score"]
    total += 1

print("Score: {}".format(score * 1.0 / total))
