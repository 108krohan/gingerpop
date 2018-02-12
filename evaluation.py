import json
import modelapi
import os
import sys

test_image_directory = "./my_test/"
all_im = os.listdir(test_image_directory)
TP = 0
FP = 0
FN = 0
total_score = 0
output_file = open("predict_output", "w")
my_results = []
count = 0
for img in all_im:
    #sys.stdout = output_file
    result = modelapi.predict(test_image_directory+img)
    

    #sys.stdout = sys.__stdout__
    #print(img, result)
    my_results.append([img, result])
    if count % 20 == 0 :
        unicodes = [chr(i) for i in result] 
        print([img, unicodes])
    count += 1

with open('my_save_file', 'w') as fw:
    fw.write(json.dumps(my_results))
