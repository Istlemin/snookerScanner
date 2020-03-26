import numpy as np
import math

def ternarySearch(a,  b,  f):
	r = (math.sqrt(5)-1)/2
	x1 = b - r*(b-a)
	x2 = a + r*(b-a)
	f1 = f(x1)
	f2 = f(x2)
	for _ in range(15):
		if f1 < f2:
			b = x2
			x2 = x1
			f2 = f1
			x1 = b - r*(b-a)
			f1 = f(x1)
		else:
			a = x1
			x1 = x2
			f1 = f2
			x2 = a + r*(b-a)
			f2 = f(x2)
	return (a+b)/2


def hillClimb(start,func):
	if len(start)==0:
		return []
	v1 = ternarySearch(start[0][0],start[0][1],
	lambda p: func(([p]+hillClimb(start[1:],lambda l: func(([p]+l))))))

	return [v1]+hillClimb(start[1:],lambda l: func(([v1]+l)))
