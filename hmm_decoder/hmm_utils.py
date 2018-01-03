def convert_to_dot(filename):
    """Converts the transition matrix shown in a csv file to a dot formatted graph"""
    file = open(filename)
    lines = file.readlines()
    header = lines[0].split('\t')[1:] #header for the second one
    graph_string = ""
    for line in lines[1:]:
        print line, "graph line"
        feats = line.split('\t')
        domain = feats[0]
        print domain
        for i in range(1,len(feats)):
            print "%%%" + feats[i] + "%%%%"
            if feats[i].strip() == "1":
                graph_string+=domain + " -> " + header[i-1].strip().strip("\n") + ";\n"
    file.close()
    #print len(graph_string.split("\n"))
    print graph_string
    return graph_string
            
    
