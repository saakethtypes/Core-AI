import numpy as np
from inspect import getfullargspec

print('Differential Evolution on Eggholder Function')
f= open("17XJ1A0328.txt","w+")
def diffe(func,bounds,mutnum1,mutnum2,cp,pop_size,gens):

	args = list(getfullargspec(func))
	nargs = len(args[0])
	#generating initial population
	pop = np.random.rand(pop_size,nargs)
	print('Initial population - \n',pop)
	#defining bounds and fitness matrix 
	minb = min(bounds[0])

	maxb = max(bounds[0])
	diff = maxb-minb
	
		#denormalising
	pop_act = minb + (pop * diff)
	print(pop_act)
	fitness_mat = []
	for i in range(len(pop_act)):
		if nargs!=1:
			fitness_mat.append(func(pop_act[i][0],pop_act[i][1]))
		else :
			fitness_mat.append(func(pop_act[i]))

	#best fitness 
	min_fitness_ix = np.argmin(fitness_mat)
	min_fitness = pop_act[min_fitness_ix]

	#evolution
	for i in range(gens):

		for j in range(pop_size):
		#mutation
			target_vec = pop[j]
			choosing_vecs = list(range(len(pop)))
			choosing_vecs.pop(j)
			selected = np.random.choice(choosing_vecs, 3)
			r1,r2,r3 = selected
			mutant_vec = np.clip(pop[r1] + mutnum1*(pop[r2]-pop[r3]) + mutnum2*(pop[min_fitness_ix]-pop[r1]),0,1)
			crossover_pos = np.random.rand(2) < cp
			#crossover_pos = np.random.rand(nargs) < cp
        #trail vector
			trail_vec = np.where(crossover_pos, mutant_vec, pop[j])
			trail_vec_act = minb + (trail_vec*diff)
        #selection
			if nargs!=1:
				trail_vec_val = func(trail_vec_act[0],trail_vec_act[1])
			else:
				trail_vec_val = func(trail_vec_act)
 
			if trail_vec_val<fitness_mat[j]:
				pop_act[j] = trail_vec_val
				if nargs!=1:
					if func(trail_vec_act[0],trail_vec_act[1]) < fitness_mat[min_fitness_ix]:
						min_fitness_ix = j
						fitness_mat[j] = func(trail_vec_act[0],trail_vec_act[1])
				else:
					if func(trail_vec_act) < fitness_mat[min_fitness_ix]:
						min_fitness_ix = j
						fitness_mat[j] = func(trail_vec_act)
			else:
				pop[j] = target_vec
		print('Converged to - ',min(fitness_mat))
		conv = str(min(fitness_mat))
		#xy = [str(pop_act[min_fitness_ix][0]),str((pop_act[min_fitness_ix][1]))]
		xysp = str(pop_act[min_fitness_ix])
		gennum = str(i)
		#f.write("Generation " + gennum +" best candidate - x,y = " + xysp +"\r\n" )#+" , "+ xy[1] 
		f.write("\t"+conv+"\r\n")
		#f.write("__________________________\n")
	f.close() 
	return min(fitness_mat)

def eggfunc(x,y):
	return -(y+47)*np.sin(np.sqrt(np.abs((x/2)+(y+47))))-x*np.sin(np.sqrt(np.abs(x - (y+47))))

def crosstray(x,y):
	return -0.0001*((np.abs(np.sin(x)*np.sin(y)*np.exp(np.abs(100-((np.sqrt(x**2 + y**2)/3.14)))))+1)**0.1)

def schafer(x,y):
	return 0.5 + (np.cos(np.sin(np.abs(x**2-y**2)))**2 - 0.5)/((1+0.001*(x**2 +y**2))**2)

def sphere(x):
	val = 0
	for i in range(len(x)):
		xi = x[i]
		val = val + xi**2
	return val/len(x)

def bukin(x,y):
	return 100*np.sqrt(np.abs(y-0.01*x**2))+0.01*np.abs(x +10)

def holder(x,y):
	return -np.abs(np.sin(x)*np.cos(y)*np.exp(np.abs(1-(np.sqrt(x**2+y**2)/np.pi))))

pop_size=250
gens=200
cp =0.8
boundsegg = [(-512, 512)] * 2
boundscrosstray = [(-10,10)] * 2
boundsschafer = [(-100,100)] * 2
boundsphere = [(-100000,100000)]
boundsbukin = [(-15,15) , (-3,3)]

F = np.random.uniform(-2,2)
k = 0.5
sphereargs= [200,38,46,121,62,332]

diffe(holder,boundscrosstray,k,F,cp,pop_size,gens)
