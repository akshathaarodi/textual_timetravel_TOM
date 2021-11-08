predictions = []
answers = []
questions = []
trace_lines = []

#trace_path = "data-original/original/"
trace_path = "../../../../../../tomi_fixed/data/data-new/data-multiqs-what-sampled/"

#folder_path = "outputs_original_bow_untied/"
folder_path = "outputs-mnn3.8/"
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


agents = [
    "Oliver",
    "Ethan",
    "Liam",
    "Benjamin",
    "Lucas",
    "Alexander",
    "Jacob",
    "Mason",
    "William",
    "Hunter",
    "James",
    "Logan",
    "Owen",
    "Noah",
    "Carter",
    "Nathan",
    "Jack",
    "Aiden",
    "Jackson",
    "Jayden",
    "Emma",
    "Olivia",
    "Emily",
    "Sophia",
    "Ava",
    "Chloe",
    "Charlotte",
    "Abigail",
    "Amelia",
    "Ella",
    "Hannah",
    "Isabella",
    "Aria",
    "Lily",
    "Mia",
    "Isla",
    "Avery",
    "Elizabeth",
    "Mila",
    "Evelyn"
  ]

where_agents = []
for agent in agents:
    #where_agents.append("where is "+ agent.lower())
    where_agents.append("what is the location of "+ agent.lower())

print(questions[10])
#print("where is jack")
#print(where_agents)
correct = 0
total = 0

for i in range(0,len(predictions)):
    question = questions[i]
    answer = answers[i]
    prediction = predictions[i]
    if question in where_agents:
        total += 1
        if answer == prediction:
            correct += 1
print(correct)
print(total)
print("Accuracy ", round(100*correct/total))
