

with open('data/labels.csv') as f:
    	# creating training data
        data = f.readlines()
        one_count = 0
        zero_count = 0
        for i in range(820):
            tuple_data = data[i].strip().split(",")
            print(tuple_data)
            if int(tuple_data[1]) == 0:
                zero_count += 1
            elif int(tuple_data[1]) == 1:
                one_count += 1
            else: 
                continue
        print(zero_count)
        print(one_count)