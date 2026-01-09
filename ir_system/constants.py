import re

# Regex for tokenization
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Standard English Stopwords
STOPWORDS = set("""
a an the and or for of to in on at by with without from is are was were be been being
this that these those it its as not no yes but if then than into over under
""".split())