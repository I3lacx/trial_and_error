import numpy as np
import matplotlib.pyplot as plt

# so value 0 is 0, after it it is i = p
prob_θ = [1.0]*101

def normalize(distr):
	summed = np.sum(distr)
	distr = [x/summed for x in distr]
	return distr
	# print(distr)
	
def get_posterior(x, prior, plot=False):
	likelihood = [(i/100)**x * (1-(i/100))**(1-x) for i in range(len(prior))]
	# likelihood = normalize(likelihood)

	posterior = [likelihood[i] * prior[i] for i in range(len(prior))]
	posterior = normalize(posterior)

	if plot:
		fig, ax = plt.subplots(2,2)
		ax[0,0].plot(prior)
		ax[0,0].set_title("prior")

		ax[0,1].plot(likelihood)
		ax[0,1].set_title("likelihood")
		
		ax[1,0].plot(posterior)
		ax[1,0].set_title("posterior")
		plt.show()
	return posterior
	
prior = normalize(prob_θ)
for i in range(100):
	ε = np.random.rand()
	x = 1 if ε > 0.5 else 0
	# print("Draw x: ", x)
	# print(prior[0])
	# print(prior[100])
	if i % 1 == 0:
		prior = get_posterior(x, prior, plot=True)
	else:
		prior = get_posterior(x, prior)
# plt.plot(np.arange(0,1.01,0.01), normalized_θ)
# plt.show()
