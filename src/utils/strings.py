import re

# Removed the number/number typical of dates (eg. 12/09/2021)
def RE_TOKEN(string): return re.compile(rf"(({string})|((\w+-\d+)|(\w+-\w+)|(-?\d+(,\d{3})*(\.\d+)?))|([^\W\d]+)|(\S))", re.UNICODE)

def custom_tokenizer(text, avoid='<UNK>', token_min_len=1, token_max_len=20, lower=True):
	tokens = []
	app = RE_TOKEN(avoid).findall(text)
	for token in app:
		if isinstance(token, tuple):
			token = token[0]

		if len(token) > token_max_len or len(token) < token_min_len:
			pass

		else:
			if token!=avoid and lower:
				token = token.lower()

			tokens.append(token)

	return tokens

def pymu_custom_tokenizer(text, avoid='<UNK>', token_min_len=1, token_max_len=20, lower=True):
	tokens = []
	app = text.split(' ')
	for token in app:
		if isinstance(token, tuple):
			token = token[0]

		if len(token) > token_max_len or len(token) < token_min_len:
			pass

		else:
			if token!=avoid and lower:
				token = token.lower()

			tokens.append(token)

	return tokens

def to_representation(token):
    # replace chars and digits
    word = ''.join(['x' if character.isdigit() else 'w' if character.isalpha() else character for character in token])
    word = re.sub(r"(.)\1+", r"\1", word)
    
    # remove number sign
    founds = list(re.finditer(r"-x", word))
    if len(founds)>0:
        to_remove = [m[0] for m in founds[0].regs if (m[0] == 0 or word[m[0]-1] not in ['+', 'w', 'x'])]
        word = ''.join([el for i, el in enumerate(word) if i not in to_remove])

    return word