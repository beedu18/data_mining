from Algos import kmeans_kmedoids as kmm
from Algos import dbscan as db
import matplotlib.pyplot as plt 

def main():
    ds=[]
    file=open('datasets/clustering2_moons.txt','r')
    lines=file.readlines()
    for i in range(len(lines)):
        string=lines[i].replace('\n','').split(',')
        ds.append([])
        for s in string:
            ds[i].append(float(s))
    
    x=kmm(ds,1,2)
    kmm.makeClusters(x)
    kmm.show_mm(x)
    kmm.plot(x)

    y=db(ds,3,1.5)
    db.classify(y)
    db.plot(y)

main()