import json
import numpy as np
from pathlib import Path
import os

NUM_SCORES = 3
TOP_EMBEDDERS = 3

def run():
    main_dir = ("C:\\Users\\aless\\TLC_Research_2024\\semantic-rag-haystack\\documents\\generation_output")

    top_embedders = []

    for file_name in os.listdir(main_dir):
        with open(Path(main_dir + "\\" + file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
            score_list = []
            for i in data:
                if (i["documents"]):
                    for sample in i["documents"]:
                        if(sample["score"]):
                            # print(sample["score"])
                            score_list.append(sample["score"])
            print("-----------Results-----------")
            print("FILE_NAME:", file_name)
            final_scores = np.partition(score_list, -NUM_SCORES)[-NUM_SCORES:] # finds the top NUM_SCORES values
            print("Scores:",final_scores)
            if(final_scores.size > 0):
                print("Max:", final_scores.max())
                print("Min:", final_scores.min())
                print("Mean:", final_scores.mean())
                print("Std:", final_scores.std())
                print("------------------------------\n")
                top_embedders.append([file_name, final_scores.max(), final_scores.min(),
                                    final_scores.mean(), final_scores.std()])
            else:
                print("NO DATA COULD BE READ FROM THIS FILE.\n")

    top_embedders.sort(key = lambda x: x[3], reverse=True) #sorts by mean value
    print(f"TOP {TOP_EMBEDDERS} EMBEDDERS:")
    for embedder in top_embedders[:TOP_EMBEDDERS]:
        print("FILE_NAME:", embedder[0])
        print("Max:", embedder[1])
        print("Min:", embedder[2])
        print("Mean:", embedder[3])
        print("Std:", embedder[4], "\n")

if __name__ == '__main__':
    run()