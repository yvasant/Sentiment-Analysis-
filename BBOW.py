def BBOW(p, data ,lines,st):
    quote=""
    finalfile = open(st + '.txt' , 'w')
    for i in range(data[0].shape[0]):
        quote = (p.sub(r':1', lines[i]))
        finalfile.write(quote)
        finalfile.write('\n')
    finalfile.close()

