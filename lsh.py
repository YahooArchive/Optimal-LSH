#! /usr/bin/python

'''
Copyright (c) 2011, Yahoo! Inc.
All rights reserved.

Redistribution and use of this software in source and binary forms, 
with or without modification, are permitted provided that the following 
conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of Yahoo! Inc. nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of Yahoo! Inc.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY 
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

# Python implementation of Andoni's e2LSH.  This version is fast because it
# uses Python hashes to implement the buckets.  The numerics are handled 
# by the numpy routine so this should be close to optimal in speed (although
# there is no control of the hash tables layout in memory.)

# This file implements the following classes
#	lsh - the basic projection algorithm (on k-dimensional hash)
#	index - a group of L lsh hashes
#	TestDataClass - a generic class for handling the raw data
 
# To use
#	Call this routine with the -histogram flag to create some random 
#		test data and to compute the nearest-neighbor distances
#	Load the .distance file that is produced into Matlab and compute
#		the d_nn and d_any histograms from the first and second columns
#		of the .distance data.
#	Use these histograms (and their bin positions) as input to the 
#		Matlab ComputeMPLSHParameters() routine. 
#	This gives you the optimum LSH parameters.  You can use these
#		values directly as parameters to this code.
#	You can use the -ktest, -ltest and -wtest flags to test the 
#		parameters.

# Prerequisites: Python version 2.6 (2.5 might work) and NumPy

# By Malcolm Slaney, Yahoo! Research

import random, numpy, pickle, os, operator, traceback, sys, math, time
import itertools					# For Multiprobe

#######################################################################
# Note, the data is always a numpy array of size Dx1.
#######################################################################

# This class just implements k-projections into k integers
# (after quantization) and then reducing that integer vector
# into a T1 and T2 hash.  Data can either be entered into a
# table, or retrieved.

class lsh:
	'''This class implements one k-dimensional projection, the T1/T2 hashing
	and stores the results in a table for later retrieval.  Input parameters
	are the bin width (w, floating point, or float('inf') to get binary LSH), 
	and the number of projections to compute for one table entry (k, an integer).'''
	def __init__(self, w, k):
		self.k = k	# Number of projections
		self.w = w	# Bin width
		self.projections = None
		self.buckets = {}

	# This only works for Python >= 2.6
	def sizeof(self):
		'''Return how much storage is needed for this object. In bytes
		'''
		return sys.getsizeof(self.buckets) + \
			sys.getsizeof(self.projections) + \
			sys.getsizeof(self)
		
	# Create the random constants needed for the projections.
	# Can't do this until we see some data, so we know the 
	# diementionality.
	def CreateProjections(self, dim):
		self.dim = dim
		# print "CreateProjections: Creating projection matrix for %dx%d data." % (self.k, self.dim)
		self.projections = numpy.random.randn(self.k, self.dim)
		self.bias = numpy.random.rand(self.k, 1)
		if 0:
			print "Dim is", self.dim
			print 'Projections:\n', self.projections
			# print 'T1 hash:\n', self.t1hash
			# print 'T2 hash:\n', self.t2hash
		if 0:
			# Write out the project data so we can check it's properties.
			# Should be Gaussian with mean of zero and variance of 1.
			fp = open('Projections.data', 'w')
			for i in xrange(0,self.projections.shape[0]):
				for j in xrange(0,self.projections.shape[1]):
					fp.write('%g ' % self.projections[i,j])
				fp.write('\n')
			
	# Compute the t1 and t2 hashes for some data.  Doing it this way 
	# instead of in a loop, as before, is 10x faster.  Thanks to Anirban
	# for pointing out the flaw.  Not sure if the T2 hash is needed since
	# our T1 hash is so strong.
	debugFP = None
	firstTimeCalculateHashes = False		# Change to false to turn this off
	infinity = float('inf')					# Easy way to access this flag
	def CalculateHashes(self, data):
		'''Multiply the projection data (KxD) by some data (Dx1), 
		and quantize'''
		if self.projections == None:
			self.CreateProjections(len(data))
		bins = numpy.zeros((self.k,1), 'int')
		if lsh.firstTimeCalculateHashes:
			print 'data = ', numpy.transpose(data)
			print 'bias = ', numpy.transpose(self.bias)
			print 'projections = ', 
			for i in range(0, self.projections.shape[0]):
				for j in range(0, self.projections.shape[1]):
					print self.projections[i][j], 
				print
			# print 't1Hash = ', self.t1hash
			# print 't2Hash = ', self.t2hash
			firstTimeCalculateHashes = False
			print "Bin values:", self.bias + \
							numpy.dot(self.projections, data)/self.w
			print "Type of bins:", type(self.bias + \
							numpy.dot(self.projections, data)/self.w)
		if 0:
			if lsh.debugFP == None:
				print "Opening Projections file"
				lsh.debugFP = open('Projections.data', 'w')
			d = self.bias + numpy.dot(self.projections, data)/self.w
			for i in xrange(0, len(d)):
				lsh.debugFP.write('%g\n' % d[i])
			lsh.debugFP.write('\n')
			lsh.debugFP.flush()
		if self.w == lsh.infinity:
			# Binary LSH
			bins[:] = (numpy.sign(numpy.dot(self.projections, data))+1)/2.0
		else:
			proj_trans = numpy.dot(self.projections, data)/self.w
			bins[:] = numpy.floor(self.bias + proj_trans[None].T)	# proj_trans[None] is a dirty ugly hack to turn your 1D vector into a 2D array and then transpose it so that dimensions of self.bias and the rest are in aggreement
		t1 = self.ListHash(bins)
		t2 = self.ListHash(bins[::-1])		# Reverse data for second hash
		return t1, t2
		
	# Input: A Nx1 array (of integers)
	# Output: A 28 bit hash value.
	# From: http://stackoverflow.com/questions/2909106/
	#	python-whats-a-correct-and-good-way-to-implement-hash/2909572#2909572
	def ListHash(self, d):
		# return str(d).__hash__()		# Good for testing, but not efficient
		if d == None or len(d) == 0:
			return 0
		# d = d.reshape((d.shape[0]*d.shape[1]))
		value = d[0, 0] << 7
		for i in d[:,0]:
			value = (101*value + i)&0xfffffff
		return value	
	
	# Just a debug version that returns the bins too.
	def CalculateHashes2(self, data):
		if self.projections == None:
			print "CalculateHashes2: data.shape=%s, len(data)=%d" % (str(data.shape), len(data))
			self.CreateProjections(len(data))
		bins = numpy.zeros((self.k,1), 'int')
		parray = numpy.dot(self.projections, data)
		bins[:] = numpy.floor(parray/self.w  + self.bias)
		t1 = self.ListHash(bins)
		t2 = self.ListHash(bins[::-1])		# Reverse data for second hash
		# print self.projections, data, parray, bins
		# sys.exit(1)
		return t1, t2, bins, parray
		
	# Return a bunch of hashes, depending on the level of multiprobe 
	# asked for.  Each list entry contains T1, T2. This is a Python
	# iterator... so call it in a for loop.  Each iteration returns
	# a bin ID (t1,t2)
	# [Need to store bins in integer array so we don't convert to 
	# longs prematurely and get the wrong hash!]
	def CalculateHashIterator(self, data, multiprobeRadius=0):
		if self.projections == None:
			self.CreateProjections(len(data))
		bins = numpy.zeros((self.k,1), 'int')
		directVector = numpy.zeros((self.k,1), 'int')
		newProbe = numpy.zeros((self.k,1), 'int')
		if self.w == lsh.infinity:
			points = numpy.dot(self.projections, data)
			bins[:] = (numpy.sign(points)+1)/2.0
			directVector[:] = -numpy.sign(bins-0.5)
		else:
			points = numpy.dot(self.projections, data)/self.w + self.bias
			bins[:] = numpy.floor(points)
			directVector[:] = numpy.sign(points-numpy.floor(points)-0.5)
		t1 = self.ListHash(bins)
		t2 = self.ListHash(bins[::-1])
		yield (t1,t2)
		if multiprobeRadius > 0:
			# print "Multiprobe points:", points
			# print "Multiprobe bin:", bins
			# print "Multiprobe direct:", directVector
			dimensions = range(self.k)
			deltaVector = numpy.zeros((self.k, 1), 'int')	# Preallocate
			for r in range(1, multiprobeRadius+1):
				# http://docs.python.org/library/itertools.html
				for candidates in itertools.combinations(dimensions, r):
					deltaVector *= 0						# Start Empty
					deltaVector[list(candidates), 0] = 1	# Set some bits
					newProbe[:] = bins + deltaVector*directVector	# New probe
					t1 = self.ListHash(newProbe)
					t2 = self.ListHash(newProbe[::-1])		# Reverse data for second hash
					# print "Multiprobe probe:",newProbe, t1, t2
					yield (t1,t2)
	
	# Put some data into the hash bucket for this LSH projection
	def InsertIntoTable(self, id, data):
		(t1, t2) = self.CalculateHashes(data)
		if t1 not in self.buckets:
			self.buckets[t1] = {t2: [id]}
		else:
			if t2 not in self.buckets[t1]:
				self.buckets[t1][t2] = [id]
			else:
				self.buckets[t1][t2].append(id)
	
	# Find some data in the hash bucket.  Return all the ids
	# that we find for this T1-T2 pair.
	def FindXXObsolete(self, data):
		(t1, t2) = self.CalculateHashes(data)
		if t1 not in self.buckets:
			return []
		row = self.buckets[t1]
		if t2 not in row:
			return []
		return row[t2]
		
	# 
	def Find(self, data, multiprobeRadius=0):
		'''Find the points that are close to the query data.  Use multiprobe
		to also look in nearby buckets.'''
		res = []
		for (t1,t2) in self.CalculateHashIterator(data, multiprobeRadius):
			# print "Find t1:", t1
			if t1 not in self.buckets:
				continue
			row = self.buckets[t1]
			if t2 not in row:
				continue
			res += row[t2]
		return res
			
	# Create a dictionary showing all the buckets an ID appears in
	def CreateDictionary(self, theDictionary, prefix):
		for b in self.buckets:		# Over all buckets
			w = prefix + str(b)
			for c in self.buckets[b]:# Over all T2 hashes
				for i in self.buckets[b][c]:#Over ids
					if not i in theDictionary:
						theDictionary[i] = [w]
					else:
						theDictionary[i] += w
		return theDictionary


	# Print some stats for these lsh buckets
	def StatsXXX(self):
		maxCount = 0; sumCount = 0; 
		numCount = 0; bucketLens = [];
		for b in self.buckets:
			for c in self.buckets[b]:
				l = len(self.buckets[b][c])
				if l > maxCount: 
					maxCount = l
					maxLoc = (b,c)
					# print b,c,self.buckets[b][c]
				sumCount += l
				numCount += 1
				bucketLens.append(l)
		theValues = sorted(bucketLens)
		med = theValues[(len(theValues)+1)/2-1]
		print "Bucket Counts:"
		print "\tTotal indexed points:", sumCount
		print "\tT1 Buckets filled: %d/%d" % (len(self.buckets), 0)
		print "\tT2 Buckets used: %d/%d" % (numCount, 0)
		print "\tMaximum T2 chain length:", maxCount, "at", maxLoc
		print "\tAverage T2 chain length:", float(sumCount)/numCount
		print "\tMedian T2 chain length:", med
	
	def HealthStats(self):
		'''Count the number of points in each bucket (which is currently
		a function of both T1 and T2)'''
		maxCount = 0; numCount = 0; totalIndexPoints = 0; 
		for b in self.buckets:
			for c in self.buckets[b]:
				l = len(self.buckets[b][c])
				if l > maxCount: 
					maxCount = l
					maxLoc = (b,c)
					# print b,c,self.buckets[b][c]
				totalIndexPoints += l
				numCount += 1
		T1Buckets = len(self.buckets)
		T2Buckets = numCount
		T1T2BucketAverage = totalIndexPoints/float(numCount)
		T1T2BucketMax = maxCount
		return (T1Buckets, T2Buckets, T1T2BucketAverage, T1T2BucketMax)
		
	# Get a list of all IDs that are contained in these hash buckets
	def GetAllIndices(self):
		theList = []
		for b in self.buckets:
			for c in self.buckets[b]:
				theList += self.buckets[b][c]
		return theList

	# Put some data into the hash table, see how many collisions we get.
	def Test(self, n):
		self.buckets = {}
		self.projections = None
		d = numpy.array([.2,.3])
		for i in range(0,n):
			self.InsertIntoTable(i, d+i)
		for i in range(0,n):
			r = self.Find(d+i)
			matches = sum(map(lambda x: x==i, r))
			if matches == 0:
				print "Couldn't find item", i
			elif matches == 1:
				pass
			if len(r) > 1: 
				print "Found big bin for", i,":", r
	

# Put together several LSH projections to form an index.  The only 
# new parameter is the number of groups of projections (one LSH class
# object per group.)
class index:
	def __init__(self, w, k, l):
		self.k = k; 
		self.l = l
		self.w = w
		self.projections = []
		self.myIDs = []
		for i in range(0,l):	# Create all LSH buckets
			self.projections.append(lsh(w, k))

	# Only works for Python > 2.6
	def sizeof(self):
		'''Return the sizeof this index in bytes.
		'''
		return sum(p.sizeof() for p in self.projections) + \
			sys.getsizeof(self)

	# Replace id we are given with a numerical id.  Since we are going 
	# to use the ID in L tables, it is better to replace it here with
	# an integer.  We store the original ID in an array, and return it 
	# to the user when we do a find().
	def AddIDToIndex(self, id):
		if type(id) == int:
			return id				# Don't bother if already an int
		self.myIDs.append(id)
		return len(self.myIDs)-1
	
	def FindID(self, id):
		if type(id) != int or id < 0 or id >= len(self.myIDs):
			return id
		return self.myIDs[id]
	
	# Insert some data into all LSH buckets
	def InsertIntoTable(self, id, data):
		intID = self.AddIDToIndex(id)
		for p in self.projections:
			p.InsertIntoTable(intID, data)

	def FindXXObsolete(self, data):
		'''Find some data in all the LSH buckets. Return a list of
		data's id and bucket counts'''
		items = [p.Find(data) for p in self.projections]
		results = {}
		for itemList in items: 
			for item in itemList:
				if item in results:			# Much faster without setdefault
					results[item] += 1
				else:
					results[item] = 1
		s = sorted(results.items(), key=operator.itemgetter(1), \
			reverse=True)
		return [(self.FindID(i),c) for (i,c) in s]
	
	def Find(self, queryData, multiprobeR=0):
		'''Find some data in all the LSH tables.  Use Multiprobe, with 
		the given radius, to search neighboring buckets.  Return a list of
		results.  Each result is a tuple consisting of the candidate ID
		and the number of times it was found in the index.'''
		results = {}
		for p in self.projections:
			ids = p.Find(queryData, multiprobeR)
			# print "Got back these IDs from p.Find:", ids
			for id in ids:
				if id in results:
					results[id] += 1
				else:
					results[id] = 1
		s = sorted(results.items(), key=operator.itemgetter(1), \
			reverse=True)
		return [(self.FindID(i),c) for (i,c) in s]
		
	def FindExact(self, queryData, GetData, multiprobeR=0):
		'''Return a list of results sorted by their exact 
		distance from the query.  GetData is a function that
		returns the original data given its key.  This function returns
		a list of results, each result has the candidate ID and distance.'''
		s = self.Find(queryData, multiprobeR)
		# print "Intermediate results are:", s
		d = map(lambda (id,count): (id,((GetData(id)-queryData)**2).sum(), \
				count), s)
		s = sorted(d, key=operator.itemgetter(1))
		return [(self.FindID(i),d) for (i,d,c) in s]
	
	# Put some data into the hash tables.
	def Test(self, n):
		d = numpy.array([.2,.3])
		for i in range(0,n): 
			self.InsertIntoTable(i, d+i)
		for i in range(0,n):
			r = self.Find(d+i)
			print r
	
	# Print the statistics of each hash table.
	def Stats(self):
		for i in range(0, len(self.projections)):
			p = self.projections[i]
			print "Buckets", i, 
			p.Stats()

	# Get al the IDs that are part of this index.  Just check one hash
	def GetAllIndices(self):
		if self.projections and len(self.projections) > 0:
			p = self.projections[0]
			return p.GetAllIndices()
		return None
			
	# Return the buckets (t1 and t2 hashes) associated with a data point
	def GetBuckets(self, data):
		b = []
		for p in self.projections:
			( t1, t2, bins, parray) = p.CalculateHashes2(data)
			print "Bucket:", t1, t2, bins, parray
			b += (t1, t2)
		return b
	
	# 
	def DictionaryPrefix(self, pc):
		prefix = 'W'
		prefixes = 'abcdefghijklmnopqrstuvwxyz'
		while pc > 0:	# Create unique ID for theis bucket
			prefix += prefixes[pc%len(prefixes)]
			pc /= len(prefixes)
		return prefix
		
	# Create a list ordered by ID listing which buckets are used for each ID
	def CreateDictionary(self):
		theDictionary = {}
		pi = 0
		for p in self.projections:
			prefix = self.DictionaryPrefix(pi)
			theDictionary = p.CreateDictionary(theDictionary,\
				prefix)
			pi += 1
		return theDictionary
	
	# Find the bucket ids that best correspond to this piece of data.
	def FindBuckets(self, data):
		theWords = []
		pi = 0
		for p in self.projections:
			prefix = self.DictionaryPrefix(pi)
			( t1, t2, bins, parray) = p.CalculateHashes2(data)
			word = prefix + str(t1)
			theWords += [word]
			pi += 1
		return theWords
	
# Save an LSH index to a pickle file.
def SaveIndex(filename, ind):
	try:
		fp = open(filename, 'w')
		pickle.dump(ind, fp)
		fp.close()
		statinfo = os.stat(filename,)
		if statinfo:
			print "Wrote out", statinfo.st_size, "bytes to", \
				filename
	except:
		print "Couldn't pickle index to file", filename
		traceback.print_exc(file=sys.stderr)

# Read an LSH index from a pickle file.	
def LoadIndex(filename):
	if type(filename) == str:
		try:
			fp = open(filename, 'r')
		except:
			print "Couldn't open %s to read LSH Index" % (filename)
			return None
	else:
		fp = filename
	try:
		ind = pickle.load(fp)
		fp.close()
		return ind
	except:
		print "Couldn't read pickle file", filename
		traceback.print_exc(file=sys.stderr)

	


class TestDataClass:
	'''A bunch of routines used to generate data we can use to test
	this LSH implementation.'''
	def __init__(self):
		self.myData = None
		self.myIndex = None
		self.nearestNeighbors = {}		# A dictionary pointing to IDs
	
	def LoadData(self, filename):
		'''Load data from a flat file, one line per data point.'''
		lineCount = 0
		try: 
			fp = open(filename)
			if fp:
				for theLine in fp:			# Count lines in file
					if theLine == '':
						break
					lineCount += 1
				dim = len(theLine.split())	# Allocate the storage array
				self.myData = numpy.zeros((dim, lineCount))
				fp.seek(0,0)				# Go back to beginning of file
				lineCount = 0
				for theLine in fp:			# Now load the data
					data = [float(i) for i in theLine.split()]
					self.myData[:,lineCount] = data
					lineCount += 1
				fp.close()
			else:
				print "Can't open %s to LoadData()" % filename
		except:
			print "Error loading data from %s in TestDataClass.LoadData()" \
				% filename
			traceback.print_exc(file=sys.stderr)
		print "self.myData has %d lines and is:" % lineCount, self.myData
				
	def SaveData(self, filename):
		'''Save this data in a flat file.  One line per data point.'''
		numDims = self.NumDimensions()
		try:
			fp = open(filename, 'w')
			if fp:
				for i in xrange(0, self.NumPoints()):
					data = self.RetrieveData(i).reshape(numDims)
					fp.write(' '.join([str(d) for d in data]) + '\n')
				fp.close()
				return
		except:
			pass
		sys.stderr.write("Can't write test data to %s\n" % filename)
		
	def CreateIndex(self, w, k, l):
		'''Create an index for the data we have in our database.  Inputs are
		the LSH parameters: w, k and l.'''
		self.myIndex = index(w, k, l)
		itemCount = 0
		tic = time.clock()
		for itemID in self.IterateKeys():
			features = self.RetrieveData(itemID)
			if features != None:
				self.myIndex.InsertIntoTable(itemID, features)
				itemCount += 1
		print "Finished indexing %d items in %g seconds." % \
			(itemCount, time.clock()-tic)
		sys.stdout.flush()
	
	def RetrieveData(self, id):
		'''Find a point in the array of data.'''
		id = int(id)						# Key in this base class is an int!
		if id < self.myData.shape[1]:
			return self.myData[:,id:id+1]
		return None
		
	def NumPoints(self):
		'''How many data point are in this database?'''
		return self.myData.shape[1]
		
	def NumDimensions(self):
		'''What is the dimensionality of the data?'''
		return self.myData.shape[0]
		
	def GetRandomQuery(self):
		'''Pick a random query from the dataset.  Return a key.'''
		return random.randrange(0,self.NumPoints())	# Pick random query
		
	def FindNearestNeighbors(self, count):
		'''Exhaustive search for count nearest-neighbor results.
		Save the results in a dictionary.'''
		numPoints = self.NumPoints()
		self.nearestNeighbors = {}
		for i in xrange(0,count):
			qid = self.GetRandomQuery()				# Pick random query
			qData = self.RetrieveData(qid)				# Find it's data
			nearestDistance2 = None
			nearestIndex = None
			for id2 in self.IterateKeys():
				if qid != id2:
					d2 = ((self.RetrieveData(id2)-qData)**2).sum()
					if id == -1:					# Debugging
						print qid, id2, qData, self.RetrieveData(id2), d2
					if nearestDistance2 == None or d2 < nearestDistance2:
						nearestDistance2 = d2
						nearestIndex = id2
			self.nearestNeighbors[qid] = \
				(nearestIndex, math.sqrt(nearestDistance2))
			if qid == -1:
				print qid, nearestIndex, math.sqrt(nearestDistance2)
				sys.stdout.flush()
	
	def SaveNearestNeighbors(self, filename):
		'''Save the nearest neighbor dictionary in a file.  Each line
		of the file contains the query key, the distance to the nearest
		neighbor, and the NN key.'''
		if filename.endswith('.gz'):
			import gzip
			fp = gzip.open(filename, 'w')
		else:
			fp = open(filename, 'w')
		if fp:
			for (query,(nn,dist)) in self.nearestNeighbors.items():
				fp.write('%s %g %s\n' % (str(query), dist, str(nn)))
			fp.close()
		else:
			print "Can't open %s to write nearest-neighbor data" % filename
	
	def LoadNearestNeighbors(self, filename):
		'''Load a file full of nearest neighbor data.'''
		self.nearestNeighbors = {}
		if filename.endswith('.gz'):
			import gzip
			fp = gzip.open(filename, 'r')
		else:
			fp = open(filename, 'r')
		if fp:
			print "Loading nearest-neighbor data from:", filename
			for theLine in fp:
				(k,d,nn) = theLine.split()
				if type(self.myData) == numpy.ndarray: # Check for array indices
					k = int(k)
					nn = int(nn)
					if k < self.NumPoints() and nn < self.NumPoints():
						self.nearestNeighbors[k] = (nn,float(d))
				elif k in self.myData and nn in self.myData:	# dictionary index
					self.nearestNeighbors[k] = (nn,float(d))
			fp.close()
			print " Loaded %d items into the nearest-neighbor dictionary." % len(self.nearestNeighbors)
		else:
			print "Can't open %s to read nearest neighbor data." % filename
					
	def IterateKeys(self):
		'''Iterate through all possible keys in the dataset.'''
		for i in range(self.NumPoints()):
			yield i
	
	def FindMedian(self):
		numDim = self.NumDimensions()
		numPoints = self.NumPoints()
		oneColumn = numpy.zeros((numPoints))
		medians = numpy.zeros((numDim))
		for d in xrange(numDim):
			rowNumber = 0
			for k in self.IterateKeys():
				oneData = self.RetrieveData(k)
				oneColumn[rowNumber] = oneData[d]
				rowNumber += 1
			m = numpy.median(oneColumn, overwrite_input=True)
			medians[d] = m
		return medians
		
	def ComputeDistanceHistogram(self, fp = sys.stdout):
		'''Calculate the nearest-neighbor and any-neighbor distance
		histograms needed for the LSH Parameter Optimization.  For
		a number of random query points, print the distance to the 
		nearest neighbor, and to any random neighbor.  This becomes 
		the input for the parameter optimization routine.  Enhanced
		to also print the NN binary projections.'''
		numPoints = self.NumPoints()
		# medians = self.FindMedian()		# Not used now, but useful for binary quantization
		print "Pulling %d items from the NearestNeighbors list for ComputeDistanceHistogram" % \
			len(self.nearestNeighbors.items())
		for (queryKey,(nnKey,nnDist)) in self.nearestNeighbors.items():
			randKey = self.GetRandomQuery()
			
			queryData = self.RetrieveData(queryKey)
			nnData = self.RetrieveData(nnKey)
			randData = self.RetrieveData(randKey)
			if len(queryData) == 0 or len(nnData) == 0:			# Missing, probably because of subsampling
				print "Skipping %s/%s because data is missing." % (queryKey, nnKey)
				continue
			anyD2 = ((randData-queryData)**2).sum()
			
			projection = numpy.random.randn(1, queryData.shape[0])
			# print "projection:", projection.shape
			# print "queryData:", queryData.shape
			# print "nnData:", nnData.shape
			# print "randData:", randData.shape
			queryProj = numpy.sign(numpy.dot(projection, queryData))
			nnProj = numpy.sign(numpy.dot(projection, nnData))
			randProj = numpy.sign(numpy.dot(projection, randData))
			
			# print 'CDH:', queryProj, nnProj, randProj
			fp.write('%g %g %d %d\n' % \
				(nnDist, math.sqrt(anyD2), \
				 queryProj==nnProj, queryProj==randProj))
			fp.flush()			
		
	def ComputePnnPany(self, w, k, l, multiprobe=0):
		'''Compute the probability of Pnn and Pany for a given index size.
		Create the desired index, populate it with the data, and then measure
		the NN and ANY neighbor retrieval rates.
		Return 
			the pnn rate for one 1-dimensional index (l=1),
			the pnn rate for an l-dimensional index, 
			the pany rate for one 1-dimensional index (l=1), 
			and the pany rate for an l-dimensional index
			the CPU time per query (seconds)'''
		numPoints = self.NumPoints()
		numDims = self.NumDimensions()
		self.CreateIndex(w, k, l)			# Put data into new index
		cnn  = 0; cnnFull  = 0
		cany = 0; canyFull = 0
		queryCount = 0							# Probe the index
		totalQueryTime = 0
		startRecallTestTime = time.clock()
		# print "ComputePnnPany: Testing %d nearest neighbors." % len(self.nearestNeighbors.items())
		for (queryKey,(nnKey,dist)) in self.nearestNeighbors.items():
			queryData = self.RetrieveData(queryKey)
			if queryData == None or len(queryData) == 0:
				print "Can't find data for key %s" % str(queryKey)
				sys.stdout.flush()
				continue
			startQueryTime = time.clock()	# Measure CPU time
			matches = self.myIndex.Find(queryData, multiprobe)
			totalQueryTime += time.clock() - startQueryTime
			for (m,c) in matches:
				if nnKey == m:				# See if NN was found!!!
					cnn += c
					cnnFull += 1
				if m != queryKey:			# Don't count the query
					cany += c
			canyFull += len(matches)-1		# Total candidates minus 1 for query
			queryCount += 1
			# Some debugging for k curve.. print individual results
			# print "ComputePnnPany Debug:", w, k, l, len(matches), numPoints, cnn, cnnFull, cany, canyFull
		recallTestTime = time.clock() - startRecallTestTime
		print "Tested %d NN queries in %g seconds." % (queryCount, recallTestTime)
		sys.stdout.flush()
		if queryCount == 0:
			queryCount = 1					# To prevent divide by zero
		perQueryTime = totalQueryTime/queryCount
		print "CPP:", cnn, cnnFull, cany, canyFull
		print "CPP:",  cnn/float(queryCount*l), cnnFull/float(queryCount), \
			cany/float(queryCount*l*numPoints), canyFull/float(queryCount*numPoints), \
			perQueryTime, numDims
		return cnn/float(queryCount*l), cnnFull/float(queryCount), \
			cany/float(queryCount*l*numPoints), canyFull/float(queryCount*numPoints), \
			perQueryTime, numDims

	def ComputePnnPanyCurve(self, wList = .291032, multiprobe=0):
			if type(wList) == float or type(wList) == int:
				wList = [wList*10**((i-10)/10.0) for i in range(0,21)]
			for w in wList:
				(pnn, pnnFull, pany, panyFull, queryTime, numDims) = self.ComputePnnPany(w, 1, 10, multiprobe)
				if w == wList[0]:
					print "# w pnn pany queryTime"
				print "PnnPany:", w, multiprobe, pnn, pany, queryTime
				sys.stdout.flush()

	def ComputeKCurve(self, kList, w = .291032, r=0):
		'''Compute the number of ANY neighbors as a function of
		k.  Should go down exponentially.'''
		numPoints = self.NumPoints()
		l = 10
		for k in sorted(list(kList)):
			(pnn, pnnFull, pany, panyFull, queryTime, numDims) = self.ComputePnnPany(w, k, l, r)
			print w, k, l, r, pnn, pany, pany*numPoints, queryTime
			sys.stdout.flush()

	def ComputeLCurve(self, lList, w = 2.91032, k=10, r=0):
		'''Compute the probability of nearest neighbors as a function
		of l.'''
		numPoints = self.NumPoints()
		firstTime = True
		for l in sorted(list(lList)):
			(pnn, pnnFull, pany, panyFull, queryTime, numDims) = self.ComputePnnPany(w, k, l, r)
			if firstTime:
				print "# w k l r pnnFull, panyFull panyFull*N queryTime"
				firstTime = False
			print w, k, l, r, pnnFull, panyFull, panyFull*numPoints, queryTime
			sys.stdout.flush()

	
	
class RandomTestData(TestDataClass):
	'''Generate uniform random data points between -1 and 1.'''
	def CreateData(self, numPoints, dim):
		self.myData = (numpy.random.rand(dim, numPoints)-.5)*2.0

class HyperCubeTestData(TestDataClass):
	'''Create a hypercube of data.  All points are in the corners'''
	def CreateData(self, numDim, noise = None):
		numPoints = 2**numDim
		self.myData = numpy.zeros((numPoints, numDim))
		for i in range(0,numPoints):
			for b in range(0,numDim):
				if (2**b) & i:
					self.myData[b, i] = 1.0
		if noise != None:
			self.myData += (numpy.random.rand(numDim, numPoints)-.5)*noise

class RegularTestData(TestDataClass):
	'''Fill the 2-D test array with a regular grid of points between -1 and 1'''
	def CreateData(self, numDivs):
		self.myData = numpy.zeros(((2*numDivs+1)**2,2))
		i = 0
		for x in range(-numDivs, numDivs+1):
			for y in range(-numDivs, numDivs+1):
				self.myData[0, i] = x/float(divs)
				self.myData[1, i] = y/float(divs)
				i += 1
	
				

# Use Dimension Doubling to measure the dimensionality of a random
# set of data.  Generate some data (either random Gaussian or a grid)
# Then count the number of points that fall within the given radius of this 
# query.
def XXXTestDimensionality2():
	binWidth = .5
	if True:
		numPoints = 100000
		myTestData = TestDataClass(numPoints, 3)	
	else:
		myTestData = RegularTestData(100)
		numPoints = myTestData.NumPoints
	k = 4; l = 2; N = 1000
	myTestIndex = index(binWidth, k, l, N)
	for i in range(0,numPoints):
		myTestIndex.InsertIntoTable(i, myTestData.RetrieveData(i))
	rBig = binWidth/8.0
	rSmall = rBig/2.0
	cBig = 0.0; cSmall = 0.0
	for id in random.sample(ind.GetAllIndices(), 2):
		qp = FindLSHTestData(id)
		cBig += myTestIndex.CountInsideRadius(qp, myTestData.FindData, rBig)
		cSmall += myTestIndex.CountInsideRadius(qp, myTestData.FindData, rSmall)
	if cBig > cSmall and cSmall > 0:
		dim = math.log(cBig/cSmall)/math.log(rBig/rSmall)
	else:
		dim = 0
	print cBig, cSmall, dim
	return ind

                   
# Generate some 2-dimensional data, put it into an index and then
# show the points retrieved.  This is all done as a function of number
# of projections per bucket, number of buckets to use for each index, and
# the number of LSH bucket (the T1 size).  Write out the data so we can
# plot it (in Matlab)
def GraphicalTest(k, l, N):
	numPoints = 1000
	myTestData = TestDataClass(numPoints, 3)	
	ind = index(.1, k, l, N)
	for i in range(0,numPoints):
		ind.InsertIntoTable(i, myTestData.RetrieveData(i))
	i = 42
	r = ind.Find(data[i,:])
	fp = open('lshtestpoints.txt','w')
	for i in range(0,numPoints):
		if i in r: 
			c = r[i]
		else:
			c = 0
		fp.write("%g %g %d\n" % (data[i,0], data[i,1], c))
	fp.close()
	return r
		

			
def SimpleTest():
	import time
	dim = 250
	numPoints = 10000
	myTestData = RandomTestData()
	myTestData.CreateData(numPoints,dim)
	myTestIndex = index(w=.4, k=10, l=10, N=numPoints)
	startLoad = time.clock()
	for id in myTestData.IterateKeys():
		data = myTestData.RetrieveData(id)
		myTestIndex.InsertIntoTable(id, data)
	endLoad = time.clock()
	print "Time to load %d points is %gs (%gms per point)" % \
		(numPoints, endLoad-startLoad, (endLoad-startLoad)/numPoints*1000.0)

	startRecall = time.clock()
	resCount = 0
	resFound = 0
	for id in myTestData.IterateKeys():
		query = myTestData.RetrieveData(id)
		res = myTestIndex.Find(query)
		if not res == None and len(res) > 0:
			resFound += 1
		if not res == None:
			resCount += len(res)
	endRecall = time.clock()
	print "Time to recall %d points is %gs (%gms per point" % \
		(numPoints, endRecall-startRecall, (endRecall-startRecall)/numPoints*1000.0)
	print "Found a recall hit all but %d times, average results per query is %g" % \
		(numPoints-resFound, resCount/float(numPoints))
			
			

	
def OutputAllProjections(myTestData, myTestIndex, filename):
	'''Calculate and output all the projected data for an index.'''
	lshProjector = myTestIndex.projections[0]
	fp = open(filename, 'w')
	for id in myTestData.IterateKeys():
		d = myTestData.RetrieveData(id)
		(t1, t2, bins, parray) = lshProjector.CalculateHashes2(d)
		fp.write('%d %d %g %g\n' % (t1, t2, bins[0][0], parray[0][0]))
	fp.close()
	
# 	Exact Optimization:
#		For 100000 5-d data use: w=2.91032 and get 0.55401 hits per bin and 0.958216 nn.
#			K=23.3372 L=2.70766 cost is 2.98756
#	Expected statistics for optimal solution:
#		Assuming K=23, L=3
#		p_nn(w) is 0.958216
#		p_any(w) is 0.55401
#		Probability of finding NN for L=1: 0.374677
#		Probability of finding ANY for L=1: 1.26154e-06
#		Probability of finding NN for L=3: 0.75548
#		Probability of finding ANY for L=3: 3.78462e-06
#		Expected number of hits per query: 0.378462

'''
10-D data:
Mean of Python NN data is 0.601529 and std is 0.0840658.
Scaling all distances by 0.788576 for easier probability calcs.
Simple Approximation:
	For 100000 5-d data use: w=4.17052 and get 0.548534 hits per bin and 0.885004 nn.
		K=19.172 L=10.4033 cost is 20.8065
Expected statistics: for simple approximation
	Assuming K=19, L=10
	Probability of finding NN for L=1: 0.0981652
	Probability of finding ANY for L=1: 1.10883e-05
	Probability of finding NN for L=10: 0.644148
	Probability of finding ANY for L=10: 0.000110878
	Expected number of hits per query: 11.0878
Exact Optimization:
	For 100000 5-d data use: w=4.26786 and get 0.556604 hits per bin and 0.887627 nn.
		K=21.4938 L=12.9637 cost is 17.3645
Expected statistics for optimal solution:
	Assuming K=21, L=13
	p_nn(w) is 0.887627
	p_any(w) is 0.556604
	Probability of finding NN for L=1: 0.0818157
	Probability of finding ANY for L=1: 4.53384e-06
	Probability of finding NN for L=13: 0.670323
	Probability of finding ANY for L=13: 5.89383e-05
	Expected number of hits per query: 5.89383
'''

if __name__ == '__main__':
	defaultDims = 10
	defaultW = 2.91032
	defaultK = 10
	defaultL = 1
	defaultClosest = 1000
	defaultMultiprobeRadius = 0
	defaultFileName = 'testData'
	cmdName = sys.argv.pop(0)
	while len(sys.argv) > 0:
		arg = sys.argv.pop(0).lower()
		if arg == '-d':
			arg = sys.argv.pop(0)
			try:
				defaultDims = int(arg)
				defaultFileName = 'testData%03d' % defaultDims
			except:
				print "Couldn't parse new value for defaultDims: %s" % arg
			print 'New default dimensions for test is', defaultDims
		elif arg == '-f':
			defaultFileName = sys.argv.pop(0)
			print 'New file name is', defaultFileName
		elif arg == '-k':
			arg = sys.argv.pop(0)
			try:
				defaultK = int(arg)
			except:
				print "Couldn't parse new value for defaultK: %s" % arg
			print 'New default k for test is', defaultK
		elif arg == '-l':
			arg = sys.argv.pop(0)
			try:
				defaultL = int(arg)
			except:
				print "Couldn't parse new value for defaultL: %s" % arg
			print 'New default l for test is', defaultL
		elif arg == '-c':
			arg = sys.argv.pop(0)
			try:
				defaultClosest = int(arg)
			except:
				print "Couldn't parse new value for defaultClosest: %s" % arg
			print 'New default number closest for test is', defaultClosest
		elif arg == '-w':
			arg = sys.argv.pop(0)
			try:
				defaultW = float(arg)
			except:
				print "Couldn't parse new value for w: %s" % arg
			print 'New default W for test is', defaultW
		elif arg == '-r':
			arg = sys.argv.pop(0)
			try:
				defaultMultiprobeRadius = int(arg)
			except:
				print "Couldn't parse new value for multiprobeRadius: %s" % arg
			print 'New default multiprobeRadius for test is', defaultMultiprobeRadius
		elif arg == '-create':			# Create some uniform random data and find NN
			myTestData = RandomTestData()
			myTestData.CreateData(100000, defaultDims)
			myTestData.SaveData(defaultFileName + '.dat')
			print "Finished creating random data.  Now computing nearest neighbors..."
			myTestData.FindNearestNeighbors(defaultClosest)
			myTestData.SaveNearestNeighbors(defaultFileName + '.nn')
		elif arg == '-histogram':		# Calculate distance histograms
			myTestData = TestDataClass()
			myTestData.LoadData(defaultFileName + '.dat')
			myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
			fp = open(defaultFileName + '.distances', 'w')
			if fp:
				myTestData.ComputeDistanceHistogram(fp)
				fp.close()
			else:
				print "Can't open %s.distances to store NN data" % defaultFileName
		elif arg == '-sanity':
			myTestData = TestDataClass()
			myTestData.LoadData(defaultFileName + '.dat')
			print myTestData.RetrieveData(myTestData.GetRandomQuery())
			print myTestData.RetrieveData(myTestData.GetRandomQuery())
		elif arg == '-b':		# Calculate bucket probabilities
			random.seed(0)
			myTestData = TestDataClass()
			myTestData.LoadData(defaultFileName + '.dat')
			myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
			# ComputePnnPanyCurve(myData, [.291032])
			myTestData.ComputePnnPanyCurve(defaultW)
		elif arg == '-wtest':		# Calculate bucket probabilities as a function of w
			random.seed(0)
			myTestData = TestDataClass()
			myTestData.LoadData(defaultFileName + '.dat')
			myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
			wList = [defaultW*.5**-i for i in range(-10,10)]
			# wList = [defaultW*.5**-i for i in range(-3,3)]
			myTestData.ComputePnnPanyCurve(wList, defaultMultiprobeRadius)
		elif arg == '-ktest':		# Calculate bucket probabilities as a function of k
			random.seed(0)
			myTestData = TestDataClass()
			myTestData.LoadData(defaultFileName + '.dat')
			myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
			# ComputePnnPanyCurve(myData, [.291032])
			kList = [math.floor(math.sqrt(2)**k) for k in range(0,10)]
			kList = [1,2,3,4,5,6,8,10,12,14,16,18,20,22,25,30,35,40]
			myTestData.ComputeKCurve(kList, defaultW, defaultMultiprobeRadius)
		elif arg == '-ltest':		# Calculate bucket probabilities as a function of l
			random.seed(0)
			myTestData = TestDataClass()
			myTestData.LoadData(defaultFileName + '.dat')
			myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
			# ComputePnnPanyCurve(myData, [.291032])
			lList = [math.floor(math.sqrt(2)**k) for k in range(0,10)]
			lList = [1,2,3,4,5,6,8,10,12,14,16,18,20,22,25,30]
			myTestData.ComputeLCurve(lList, w=defaultW, 
				k=defaultK, r=defaultMultiprobeRadius)
		elif arg == '-timing':
			# sys.argv.pop(0)
			timingModels = []
			while len(sys.argv) > 0:
				print "Parsing timing argument", sys.argv[0], len(sys.argv)
				if sys.argv[0].startswith('-'):
					break
				try:
					(w,k,l,r,rest) = sys.argv[0].strip().split(',', 5)
					timingModels.append([float(w), int(k), int(l), int(r)])
				except:
					print "Couldn't parse %s.  Need w,k,l,r" % sys.argv[0]
				sys.argv.pop(0)
			myTestData = TestDataClass()
			myTestData.LoadData(defaultFileName + '.dat')
			myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
			for (w, k, l, r) in timingModels:
				sys.stdout.flush()
				(pnnL1, pnn, panyL1, pany, perQueryTime, numDims) = myTestData.ComputePnnPany(w, k, l, r)
				print "Timing:", w, k, l, r, myTestData.NumPoints(), pnn, pany, perQueryTime*1000.0, numDims
		
		elif arg == '-test':		# Calculate bucket probabilities as a function of l
			random.seed(0)
			myTestData = TestDataClass()
			myTestData.LoadData(defaultFileName + '.dat')
			myTestData.LoadNearestNeighbors(defaultFileName + '.nn')
			# ComputePnnPanyCurve(myData, [.291032])
			myTestData.ComputeLCurve([defaultL], w=defaultW, k=defaultK)
		else:
			print '%s: Unknown test argument %s' % (cmdName, arg)
