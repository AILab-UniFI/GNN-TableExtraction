# THIS IS FOR POLYNOMIALS
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

def Lagrange_Polynomial(values, x):
	"""
	:param values: values (prototypes)
	:return:  weights for each embeddings

	Polynomial interpolation using Lagrange Polynomial
	f(x) = \sum_{i=0}^{N-1} y_i \prod_{j=0, j \not=i}^{N-1} \frac{x-x_j}{x_i - x_j}
	"""

	N = len(values)

	def get_weights(i):
		w = 1
		for j in range(N):
			if j != i:
				w *= (x - values[j]) / (values[i] - values[j])

		return w

	weights = [
		get_weights(i) for i in range(N)
	]

	return weights

def Linear_Interp(values, x):
	"""
	:param values: values (prototypes)(sorted from min to max)
	:return:  weights for each embeddings

	Linear interpolation
	"""
	N = len(values)

	weights = [0]*N

	if x > values[N - 1]:
		# extrapolation right
		l, r = N-1, N

	else:
		for i in range(N):
			if x < values[i]:
				if i == 0:
					# extrapolation left
					l, r = i, i+1
				else:
					l, r = i-1, i

				break


	weights[l] = 1 - (x - values[l])/(values[r]-values[l])
	weights[r] = (x - values[l])/(values[r]-values[l])

	return weights

# TODO add vanilla weighted_average methods

# values = [0.1, 1.2, 60, 300, 40000000]
# print(Lagrange_Polynomial(values, 30000000))
# [-3.415879733985816e+17, 3.492592913466299e+17, -7985179300648976.0, 313861352600615.44, 0.31640529734775136]
# print(Linear_Interp(values, 30000000))
# [0, 0, 0, 0.25000187501406257, 0.7499981249859374]

def weighted_average(values, x):
	"""
	:param values: values (prototypes)
	:return:  weights for each embeddings

	Linear interpolation
	"""

	weights = 1/np.abs(x - values)
	weights /= np.sum(weights)

	return weights

def weighted_log(x):
	"""
	:param values: values (prototypes)
	:return:  weights for each embeddings

	Linear interpolation
	"""
	if x > 1:
		x = np.log(x) + 1
	elif x < -1:
		x = -1* (np.log(np.abs(x)) + 1)

	return x

def weighted_identity(x):
	"""
	:param values: values (prototypes)
	:return:  weights for each embeddings

	Linear interpolation
	"""
	return x

def weighted_log_average(values, x):
	"""
	:param values: values (prototypes)
	:return:  weights for each embeddings

	Linear interpolation
	"""
	log_value = [weighted_log(v) for v in values]
	log_x = weighted_log(x)
	
	return weighted_average(log_value, log_x)

if __name__ == '__main__':
    values = np.array([-3000000000, 0.1, 1.2, 60, 300, 40000000])
    print(weighted_average(values, 30000000))
    # [0.00141243 0.14265512 0.14265513 0.14265541 0.14265655 0.42796537]
    print(weighted_log_average(values, 30000000))
    # [0.00645083 0.01461263 0.01554108 0.02017417 0.0229944  0.92022689]