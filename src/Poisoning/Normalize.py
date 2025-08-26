import re


# Helper function to normalize a docstring or comment
# this is used to remove the "noise" in such comments
# in order to find similarities between them.

QUOTATIONREGEX = re.compile(r"^#?\s*|^[\"']+|[\"']+$")
WHITESPACEREGEX = re.compile(r"\s+")
NEWLINEREGEX = re.compile(r"\n+")
HASHREGEX = re.compile(r"#")


def serialize(comment: str) -> str:
    # prepare a comment for serialization
    return comment.encode("unicode_escape").decode("utf-8")


def deserialize(comment: str) -> str:
    # deserialize a comment
    return comment.encode("utf-8").decode("unicode_escape")


def normalize(inputstr: str,
              removeNewLines: bool = True,
              simplyfyWhiteSpaces: bool = True) -> str:
    # clean up leading hashes and quotes
    text = QUOTATIONREGEX.sub("", inputstr)

    if removeNewLines:
        # replace newlines with spaces
        text = NEWLINEREGEX.sub(" ", text)

    # remove hashes
    text = HASHREGEX.sub(" ", text)

    if simplyfyWhiteSpaces:
        # merge spaces, tabs etc.
        text = WHITESPACEREGEX.sub(" ", text)
    text = text.strip()
    return text
