def tfidf(p,st,f):
	#p==q
	tf_data = load_svmlight_file("F:\\ACADS\\acads 6th sem\\NLP\\Assignments\\Assign 2\\" + st +"_TF.feat")
	iterator = p.finditer(f)
	doc =[0]*tf_data[0].shape[1]
	for match in iterator:
	    x=match.span()[0]+1
	    y = match.span()[1]-1
	    if x==y:
	        num = int(f[x])
	    else:
	        num = int(f[x:y])
	    doc[num]=doc[num]+1
	d = tf_data[0].shape[0]
	doc[:] = [math.log(d) if x==0 else math.log(d /x) for x in doc]
	TFIDF = tf_data[0].multiply(doc)
	dump_svmlight_file(TFIDF,tf_data[1], st + "_TFIDF.feat")
