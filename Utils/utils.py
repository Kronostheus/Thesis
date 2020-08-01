
def get_spans(labels, codes):
    limits = []
    prev = ""
    for idx, (lbl, code) in enumerate(zip(labels, codes)):
        if lbl == 'B' or (prev != code and lbl == 'O'):
            limits.append(idx)
        prev = code

    limits.append(len(labels))

    if len(limits) <= 1:
        if limits[0] == 0:
            raise Exception
        else:
            limits = [0] + limits
    elif limits[0] != 0:
        limits = [0] + limits

    return limits
