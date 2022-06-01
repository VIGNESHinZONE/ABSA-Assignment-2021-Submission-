import re


def clean_dataframe(df, text_field, phrase_field):

    punctuation = '!"#$%()*+-/:;<=>@[\\]^_`{|}~'

    df[text_field] = df[text_field].apply(lambda x: re.sub(r"http\S+", "", x))
    df[text_field] = df[text_field].apply(
        lambda x: "".join(ch for ch in x if ch not in set(punctuation)))
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].str.replace("[0-9]", " ")
    df[text_field] = df[text_field].apply(lambda x: " ".join(x.split()))

    df[phrase_field] = df[phrase_field].apply(
        lambda x: re.sub(r"http\S+", "", x))
    df[phrase_field] = df[phrase_field].apply(
        lambda x: "".join(ch for ch in x if ch not in set(punctuation)))
    df[phrase_field] = df[phrase_field].str.lower()
    df[phrase_field] = df[phrase_field].str.replace("[0-9]", " ")
    df[phrase_field] = df[phrase_field].apply(
        lambda x: " ".join(x.split()))
    return df
