
bidlabels = [1, 2, 3, 4]

def simplifyBids(bidArray):
    newbidarray = []
    newbidarray.append(bidlabels[0])
    for i in range(0, len(bidArray)-1):
        if bidArray[i+1][0]-0.05 <= bidArray[i][0] <= bidArray[i+1][0] + 0.05:
            newbidarray.append(bidlabels[1])
        elif bidArray[i][0] > bidArray[i+1][0] + 0.05:
            newbidarray.append(bidlabels[2])
        else:
            newbidarray.append(bidlabels[3])
    return newbidarray
