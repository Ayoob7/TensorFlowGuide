numberOfBuckets = 6
def hash_string(keyword, divident):
    ordTot = 0
    for e in keyword:
        ordTot += ord(e)
    return ordTot % divident

listOfList = [[],[],[],[],[],[]]

def insertInLOL(i):
    listOfList[hash_string(i,numberOfBuckets)].append(i)

insertInLOL('hello')
print(listOfList)

def search(i):
    return (listOfList[hash_string(i,numberOfBuckets)])
print(search('hello'))