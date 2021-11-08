predictions = []
answers = []
questions = []
trace_lines = []

#trace_path = "data-original/original/"
trace_path = "../../../../../../tomi_fixed/data/data-new/data-after/"

#folder_path = "outputs_original_bow_untied/"
folder_path = "outputs-mnn3.13/"
#folder_path = ""

#split = "test"
split = "val"
split_pred = ""
if split == "test":
    split_pred = "_test"

prediction_file = folder_path + "predictions" + split_pred + ".txt"
answers_file = folder_path + "answers" + split_pred + ".txt"
questions_file = folder_path +"questions" + split_pred +".txt"
#trace_file = trace_path + "fb_all_"+ split + ".trace"
trace_file = trace_path + split + ".trace"
with open(prediction_file,"r") as f:
    predictions = f.readlines()
with open(answers_file,"r") as f:
    answers = f.readlines()
with open(questions_file,"r") as f:
    questions = f.readlines()
with open(trace_file,"r") as f:
    trace_lines = f.readlines()

for i in range(0,len(predictions)):
    if predictions[i][-1] == '\n':
        predictions[i] = predictions[i][:-1]
    if answers[i][-1] == '\n':
        answers[i] = answers[i][:-1]
    if questions[i][-1] == '\n':
        questions[i] = questions[i][:-1].strip()

fcorrect = { "first_order":0, "second_order":0}
ftotal = {  "first_order":0, "second_order":0}
tcorrect = { "first_order":0, "second_order":0}
ttotal = { "first_order":0, "second_order":0}
correct = { "reality": 0,  "memory":0}
total = { "reality": 0, "memory":0}


for i in range(0,len(predictions)):
    question = questions[i]
    answer = answers[i]
    prediction = predictions[i]
    trace_word = trace_lines[i].split(",")[-2]

    if trace_word.find("memory")>-1:
        total["memory"] += 1
        if answer == prediction:
            correct["memory"] += 1
    
    if trace_word.find("reality")>-1:
        total["reality"] += 1
        if answer == prediction:
            correct["reality"] += 1

    if trace_word.find("no_tom")>-1:
#        print("no tom")
        if trace_word.find("first_order")>-1:
             ttotal["first_order"] += 1
             if answer == prediction:
                 tcorrect["first_order"] += 1
#	     else:
#                 print("The weong one is on line ",i)
#                 print("Answer is and prediction is ", answer, prediction)
        if trace_word.find("second_order")>-1:
             ttotal["second_order"] += 1
             if answer == prediction:
                 tcorrect["second_order"] += 1


    elif trace_word.find("tom")>-1:
        if trace_word.find("first_order")>-1:
             ftotal["first_order"] += 1
             if answer == prediction:
                 fcorrect["first_order"] += 1
        if trace_word.find("second_order")>-1:
             ftotal["second_order"] += 1
             if answer == prediction:
                 fcorrect["second_order"] += 1


print("Overall accuracy")
print(round(100*(correct["memory"] + correct["reality"] + tcorrect["first_order"] + fcorrect["first_order"] + tcorrect["second_order"] + fcorrect["second_order"]) / ( total["memory"] + total["reality"] + ttotal["first_order"] + ftotal["first_order"] +  ttotal["second_order"] + ftotal["second_order"]),2))

print("Memory :", round(100*(correct["memory"]) / ( total["memory"]),2))
print("Reality :", round(100*(correct["reality"]) / ( total["reality"]),2))
print("First order :",round(100*(tcorrect["first_order"] + fcorrect["first_order"]) / ( ttotal["first_order"] + ftotal["first_order"]),2))
print("Second order :",round(100*(tcorrect["second_order"] + fcorrect["second_order"]) / ( ttotal["second_order"] + ftotal["second_order"]),2))

print(correct)
print(total)

print("FB correct accuracy is", round(100*(fcorrect["first_order"] + fcorrect["second_order"]) /( ftotal["first_order"] + ftotal["second_order"]),2))
print("First Order: ", round(100*fcorrect["first_order"]/ftotal["first_order"],2))
print("Second Order: ", round(100*fcorrect["second_order"]/ftotal["second_order"],2))

print("Correct numbers")
print(fcorrect)
print("Total numbers")
print(ftotal)



print("TB report correct accuracy is", round(100*( tcorrect["first_order"] + tcorrect["second_order"])/ ( ttotal["first_order"] + ttotal["second_order"] ),2))

print("First Order: ", round(100*tcorrect["first_order"]/ttotal["first_order"],2))
print("Second Order: ", round(100*tcorrect["second_order"]/ttotal["second_order"],2))

print("Correct numbers")
print(tcorrect)
print("Total numbers")
print(ttotal)
