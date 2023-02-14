from string import ascii_lowercase


def generate():
    return [{'title': c, 'code_img': f'{c}-{c.__hash__()}'} for c in ascii_lowercase]
