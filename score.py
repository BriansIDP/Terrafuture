import json
import sys, os


infile = sys.argv[1]
with open(infile) as fin:
    data = json.load(fin)
score = 0
total = 0
pages = 0
max_score = 0
qid_to_score = {}
for datapiece in data:
    if datapiece["qid"] not in qid_to_score:
        qid_to_score[datapiece["qid"]] = []
    qid_to_score[datapiece["qid"]].append(datapiece["score"])
    pages += len(datapiece["retrieved_pages"])

for qid, scores in qid_to_score.items():
    score += sum(scores)/len(scores)
    max_score += max(scores)
    total += 1

print("Average Score: {}".format(score * 1.0 / total))
print("Max Score: {}".format(max_score * 1.0 / total))
print("Average pages: {}".format(pages / total / 8))
