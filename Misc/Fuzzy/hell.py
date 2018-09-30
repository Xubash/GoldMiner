with open("names.txt","r") as f:
    names = f.read().split("\n")

from fuzzywuzzy import process
def get_matches(query,choices):
    results = process.extract(query,choices)
    return results
get_matches("bash",names)
