import math
import numpy as np

def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor



def inter_digit(string):
    prev, i, next = string
    if prev.isdigit() and next.isdigit():
        return '.'
    return i


def number_handler(token):
	num_str = ''
	m = token
	if m:
		try:
			# remove thousands separator
			num_str = m.replace(',', '')  # eg: 1,000 -> 1000

			# remove leading zeros
			num_str = num_str.lstrip('0')  # eg: 0001 -> 1
			if num_str == '':  # eg: 0000 -> 0
				num_str = '0'
			else:
				if num_str[0] == '.':  # eg: 000.1 -> .1  -> 0.1
					num_str = '0' + num_str

			# TODO: why comment out?
			# remove excessive 0 at end of float number
			if '.' in num_str:  # eg: 0.100-> 0.1  1.100->1.1
				num_str = str(float(num_str))

			# special case
			if '/' in num_str:  # date or frac
				num_str = number_handler_help_slash(num_str)

			elif num_str.find('-', 1) >= 0:  # eg: i-386
				num_str = number_handler_help_hyphen(num_str)

		except ValueError:
			num_str = ''
			print("error--unknown format when handling number:", token)
	return num_str


def number_handler_help_slash(token):
	if '/' not in token:  # sanity check
		return ''
	tmp = token.split('/')
	# handle date, caveate: only yyyy/mm/dd or yyyy/mm  format
	year_low = 1000
	year_up = 2025
	month_low = 1
	month_up = 12
	day_low = 1
	day_up = 31

	try:
		if '' in tmp:  # ambiguous case: /07 or 2018/
			raise ValueError
		if len(tmp) == 2:  # yyyy/mm or frac

			if '.' in token:
				return str(np.float32("{0:.4f}".format(np.float32(tmp[0]) / float(tmp[1]))))

			tmp0 = int(tmp[0])
			tmp1 = int(tmp[1])
			if tmp0 > year_low and tmp0 < year_up and tmp1 > month_low and tmp1 < month_up:  # valid date format
				return [str(tmp0), '/', str(tmp1)]
			else:  # frac likely
				return str(np.float32("{0:.4f}".format(tmp0 / tmp1)))
		if len(tmp) == 3:  # yyy/mm/dd

			if '.' in token:
				raise ValueError

			tmp0 = int(tmp[0])
			tmp1 = int(tmp[1])
			tmp2 = int(tmp[2])
			if tmp0 > year_low and tmp0 < year_up and tmp1 > month_low and tmp1 < month_up and tmp2 > day_low and tmp2 < day_up:  # valid date format
				return [str(tmp0), '/', str(tmp1), '/', str(tmp2)]
			else:
				raise ValueError
	except:
		print('error--unknown format when handling slash:', token)
		return ''


def number_handler_help_hyphen(token):
	# for u-32 -> 'u', '-'. '32'
	# for -321 -> -321
	if '-' not in token:  # sanity check
		return ''
	tmp = token.split('-')
	res = []
	for i, element in enumerate(tmp):
		if i == len(tmp) - 1:
			res.append(element)
		else:
			res.append(element)
			res.append('-')
	return res


def is_numeral(token):
	try:
		num = np.float32(token)
		if num == np.float32('inf') or num == np.float32('-inf') or math.isnan(num):
			return False

		return True
	except ValueError:
		return False

def to_numeral(token):
	try:
		num = np.float32(token)
		if num == np.float32('inf') or num == np.float32('-inf') or math.isnan(num):
			return None

		return num
	except ValueError:
		return None


def to_numeral_if_possible(token):
	try:
		num = np.float32(token)
		if num == np.float32('inf') or num == np.float32('-inf') or math.isnan(num):
			return False, token

		return True, num
	except ValueError:
		return False, token
