import matplotlib.pyplot as plt

#helper Functions
def extCols(arr,col):   #returns a column of a 2D array (as a list)
   return [row[col] for row in arr]

def dist(p1,p2,flag=0):    #returns euclidean distance between points p1 and p2
    return sum([(p1[i]-p2[i])**2 for i in range(len(p1)-flag)])**0.5

def findMean(cluster):  #finds the mean of a given cluster
    points=len(cluster)
    dim=len(cluster[0])
    final=[0]*dim
    for point in cluster:
        for i in range(dim):
            final[i]+=point[i]
    for i in range(dim):
        final[i]/=points
    return final

def findMedoid(cluster,mean):    #finds the medoid of a given cluster
    medoid=cluster[0]
    Min=dist(mean,cluster[0])
    for i in range(len(cluster)):
        if dist(mean,cluster[i])<Min:
            Min=dist(mean,cluster[i])
            medoid=cluster[i]
    return medoid

def errorCheck(array):
    for i in array:
        if i!=-1:
            return 0
    return -1

def list_2_str(array):
    Str=str()
    for elem in array:
        Str+=str(elem)+","
    return Str[:len(Str)-1]

def rotate(array):
    Len=len(array)
    temp=[]
    for i in range(Len):
        array.append(array[i])
        temp.append(array[i:Len+i])
    return temp

#K-means K-medoids class
class kmeans_kmedoids:      #CLUSTERING
    def __init__(self,dataset,type_,k=3):
        self.dataset=dataset
        self.k=k
        self.means=[0]*self.k
        self.medoids=[0]*self.k
        self.prev=[0]*self.k
        self.clusters=[0]*self.k
        self.type=type_     #0==medoid, 1==mean
        for i in range(self.k):     #take first points as the initial means and medoids
            self.medoids[i]=self.dataset[i]
            self.means[i]=self.dataset[i]
        self.iters=0
        
    def makeClusters(self):
        while True:
            self.iters+=1
            for i in range(self.k):
                self.clusters[i]=[]     #clear clusters before each iteration

            for point in self.dataset:  
                temp=[]
                if self.type==0:    #make cluters based on medoids
                    for medoid in self.medoids:
                        temp.append(dist(point,medoid))
                elif self.type==1:  #make cluters based on means
                    for mean in self.means:
                        temp.append(dist(point,mean))    
                index=temp.index(min(temp))
                self.clusters[index].append(point)

            self.prev=self.means.copy() #save previous copy of means    
            for i in range(self.k):
                self.means[i]=findMean(self.clusters[i])    #update means
            
            if self.prev==self.means and self.type!=0:  #compare
                break


            if self.type==0:        #find medoid for each cluster
                self.prev=self.medoids.copy()   #save previous copy of medoids
                for i in range(self.k):
                    self.medoids[i]=findMedoid(self.clusters[i],self.means[i])  #update medoids
                
                if self.prev==self.medoids:  #compare
                    break
        
        print(f'Convergence after {self.iters} iterations')

    def showClusters(self):
        for i in range(self.k):
            print("\nCluster["+str(i+1)+"]:\n\n",self.clusters[i])
    
    def show_mm(self):
        if self.type==1:
            print("Means:")
            for i in range(self.k):
                print(self.means[i])
        elif self.type==0:
            print("Medoids:")
            for i in range(self.k):
                print(self.medoids[i])

    def plot(self):
        for cluster in self.clusters:
            plt.scatter(extCols(cluster,0),extCols(cluster,1),s=15)
        if self.type==0:
            plt.scatter(extCols(self.medoids,0),extCols(self.medoids,1),s=15,marker='x',color='black')        
        elif self.type==1:
            plt.scatter(extCols(self.means,0),extCols(self.means,1),s=15,marker='x',color='black')
        plt.show()

#dbscan class
class dbscan:       #CLUSTERING
    def __init__(self,dataset,minpts,eps):
        self.dataset=dataset
        self.minpts=minpts
        self.eps=eps
        self.clusters={}
        self.C=0    #intial number of clusters
        for i in range(len(self.dataset)):  #additional property added to points
            self.dataset[i].append(None)

    def nbhd(self,point):   #finds neighbourhood of a point.
        count=0
        pts=[]
        for pt in self.dataset:
            d=dist(point,pt,1)
            if(d<=self.eps and d!=0):
                count+=1
                pts.append(pt)
        return [count,pts]

    def classify(self):   #classifies each point in the dataset (modifies the new property added above)
        for pt in self.dataset:
            if pt[2]!=None:
                continue
            neighbors=self.nbhd(pt)
            if neighbors[0]<self.minpts:
                pt[2]='Noise'
                continue
            self.C+=1
            pt[2]=self.C
            seed=self.nbhd(pt)[1]
            for q in seed:
                if q[2]=='Noise':
                    q[2]=self.C
                if q[2]!=None:
                    continue
                q[2]=self.C 
                temp_nbh=self.nbhd(q)
                if temp_nbh[0]>=self.minpts:
                    for pt_ in temp_nbh[1]:
                        if pt_ not in seed:
                            seed.append(pt_)

    def plot(self):     #for plotting the points
        for pt in self.dataset:
            if pt[2] not in self.clusters:
                self.clusters[pt[2]]=[]     #clusters initialized as empty lists

        for pt in self.dataset:
            self.clusters[pt[2]].append(pt)   #points added to respective clusters

        for k,v in self.clusters.items():
            if k=='Noise':
                plt.scatter(extCols(v,0),extCols(v,1),marker='x',color='black',s=15)
                continue
            plt.scatter(extCols(v,0),extCols(v,1),marker='o',s=5)
        plt.show()

#apriori class
class apriori:      #ASSOCIATION RULE MINING
    def __init__(self,dataset,minsup,minconf):
        self.dataset=dataset
        self.minsup=minsup
        self.minconf=minconf
        self.items=set()
        self.N=len(self.dataset)    #total number of transactions
        Max=len(self.dataset[0])
        for row in self.dataset:
            if len(row)>Max:
                Max=len(row)
            for item in row:
                self.items.add(item)
        self.w=Max  #max width of transcation
        self.itemsetDict={}
        self.count=0

    def isFreq(self,itemset): #determines if an itemset is frequent or not
        count=0
        op=len(itemset)
        if op>self.w:
            return [False,0]
        if tuple(itemset) not in self.itemsetDict:
            self.count+=1
            # print('freq')    
            for row in self.dataset:
                if len(itemset&(set(row)))==op:
                    count+=1
            if count>=self.minsup*self.N:
                self.itemsetDict[tuple(itemset)]=[True,count]
            else:
                self.itemsetDict[tuple(itemset)]=[False,count]
        return self.itemsetDict[tuple(itemset)]

    def calcConf(self,X,Y): #calculate confidence of the rule X-->Y
        X,Y=set(X),set(Y)
        return self.isFreq(X|Y)[1]/self.isFreq(X)[1]

    def calcSupp(self,X,Y):  #calculate support of the rule X-->Y
        X,Y=set(X),set(Y)
        return self.isFreq(X|Y)[1]/self.N       

    def nextFreqItemSet(self,itemset): #returns all k+1 frequent itemsets of a k-itemset  
        if type(itemset)==str:
            itemset={itemset}
        Next=[]
        if itemset==-1:
            return -1
        else:
            for item in self.items:
                temp=itemset.copy()
                if item not in itemset:
                    temp.add(item)
                    if self.isFreq(temp)[0]:
                        Next.append(temp)
                temp=None
            return Next if len(Next)!=0 else -1

    def allFreqItemSet(self):  #returns all frequent itemsets of the transaction dataset
        temp=set()
        items=self.items
        valid=items
        while errorCheck(items)==0:
            for item in items:
                if type(item)==frozenset:
                    item=set(item)
                tempNextFreq=self.nextFreqItemSet(item)
                if tempNextFreq!=-1:
                    for i in tempNextFreq:
                        temp.add(frozenset(i))
                else:
                    temp.add(-1)
            valid|=temp
            items=temp.copy()
            temp.clear()
        temp=valid.copy()
        valid.clear()
        for item in temp:
            if type(item)!=str and item!=-1:
                valid.add(item)
        return valid

    def validRules(self,itemset):  #returns all valid rules for a given itemset
        rules=set()
        itemset=list(itemset)
        i=len(itemset)-1
        while i>=1:
            c1=self.calcConf(itemset[:i],itemset[i:])>=self.minconf
            c2=self.calcSupp(itemset[:i],itemset[i:])>=self.minsup
            if c1 and c2:    
                rules.add(list_2_str(itemset[:i])+"-->"+list_2_str(itemset[i:]))
            else:
                break
            i-=1
        return rules

    def generateRules(self):  #returns all association rules for the transaction dataset
        freq=self.allFreqItemSet()
        rules=set()
        for itemset in freq:
            Rotations=rotate(list(itemset))
            for Rotation in Rotations:
                for rule in self.validRules(Rotation):
                    rules.add(rule)
        return [rules,self.count]

#Naive Bayesian Classifier 
class NBC:      #CLASSIFICATION
    def __init__(self,dataset,attr):
        if len(attr)!=len(dataset[0])-1:
            print("Insufficient Attributes for object!\nExiting!!!")
            quit()
        self.dataset=dataset    #the training data
        self.attr=attr          #the object to classify
        self.Class=dict()       #Number of classes in dataset
        for obj in self.dataset:
            cls=obj[-1]         #extract class of the object
            if cls not in self.Class:
                self.Class[cls]=[1,None,1]  #initial count, apriori probability, temporary posteriori probability
            else:
                self.Class[cls][0]+=1
                self.Class[cls][1]=self.Class[cls][0]/len(self.dataset)

    def classify(self):
        for k,v in self.Class.items():
            for i in range(len(self.attr)):
                count=0
                for row in self.dataset:
                    if row[-1]==k and row[i]==self.attr[i]:
                        count+=1
                temp=count/self.Class[k][0]                     #marginal probability
                if temp==0:
                    temp=1/(self.Class[k][0]+len(self.Class))   #laplacian estimation if marginal probability is 0
                self.Class[k][2]*=temp                          #class conditional probability
            self.Class[k][2]*=self.Class[k][1]                  #posteriori probability

    def show(self):
        print('\nPosteriori probability:')
        for k,v in self.Class.items():
            print(f'{k} = {v[2]}')
        print('\nObject belongs to class:',max(self.Class.items(),key=lambda x:x[1][2])[0])
