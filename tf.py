def tf(p , lines ,data ,st):
    sum1 = 0
    finalfile = open(st + '_TF.txt' , 'w')
    tftext = ""
    for i in range(len(lines)):
        iterator = p.finditer(lines[i])
        for match in iterator:
            x=match.span()[0]+1
            y = match.span()[1]-1
            if x==y:
                num = int(lines[i][x])
            else:
                num = int(lines[i][x:y])
            sum1 = sum1 + num
        f = str(round((1/sum1),10))
        tftext = p.sub(r''+':'+ f,lines[i])
        finalfile.write(tftext)
        finalfile.write('\n')
        sum1 = 0
    finalfile.close()
    tfmult = load_svmlight_file("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\"+st +"_TF.txt")
    finalTF = data[0].multiply(tfmult[0])
    dump_svmlight_file(finalTF,data[1], st+ "_TF.feat")
