import random
file_data = []
file_trace = []
selected_story_ids = []

file_name = "sampledtest"
folder = 'data'
with open(folder+'/'+file_name+'.txt', "r") as f:
    file_data = f.readlines()
with open(folder+'/'+file_name+'.trace', "r") as f:
    file_trace = f.readlines()
question_lengths = []

required_questions = 1000

question_types = ['memory', 'reality','fo_tom', 'so_tom', 'fo_no_tom','so_no_tom']

first_line = file_data[0]
questions = 0
stories = []
story = []
story_ind = 0
quota = {}

for q in question_types:
    quota[q] = 0

for line in file_data:
    
    if line.split(" ")[0] == '1':
        #print("story changed",line, first_line)
        question_lengths.append(questions)
        questions = 0
        first_line = line
        stories.append(story)
        story = [] 
        
    
    if line.find("?")>-1:
        questions = questions + 1

    story.append(line)

stories.append(story)
stories = stories[1:]

trace_words = []
for i in range(0,len(file_trace)):
    line = file_trace[i]
    if line.find("where_agent") > -1:
        selected_story_ids.append(i)
    trace_words.append(line.split(",")[-2])


question_lengths.append(questions)
question_lengths = question_lengths[1:]
# print("stories are")
# print(stories)
# print("story_lengths are")
# print(question_lengths)

def get_num_questions(story):
   questions = 0
   for line in story:
       if line.find('?') > -1:
           questions = questions + 1
   return questions

def remove_excess_questions(story, total_qs):
    excess_qs = total_qs - required_questions
    while excess_qs > 0:
        story = story[:-1]
        excess_qs = excess_qs - 1
    return story

final_story = []


for i in range(0, len(stories)):
    story = stories[i]
    trace_word = trace_words[i]
    if trace_word.find("memory") > -1:
        if quota["memory"] < required_questions:
            selected_story_ids.append(i)
            quota["memory"] += 1
    if trace_word.find("reality") > -1:
        if quota["reality"] < required_questions:
            selected_story_ids.append(i)
            quota["reality"] += 1

    if trace_word.find("no_tom") > -1:
        if trace_word.find("first_order") > -1 and quota["fo_no_tom"] < required_questions:
            selected_story_ids.append(i)
            quota["fo_no_tom"] += 1
        if trace_word.find("second_order") > -1 and quota["so_no_tom"] < required_questions:
            selected_story_ids.append(i)
            quota["so_no_tom"] += 1

    elif trace_word.find("tom") > -1:
        if trace_word.find("first_order") > -1 and quota["fo_tom"] < required_questions:
            selected_story_ids.append(i)
            quota["fo_tom"] += 1
        if trace_word.find("second_order") > -1 and quota["so_tom"] < required_questions:
            selected_story_ids.append(i)
            quota["so_tom"] += 1



#Location stories have ids of what is the location of X questions
print("selected_story_ids stories are:")
print(selected_story_ids)
print("stories are")
print(len(stories))
print("story_lengths are")
print(len(selected_story_ids))
print("quota is")
print(quota)


final_story = [stories[i] for i in range(0, len(stories)) if i in selected_story_ids]
final_trace = [file_trace[i] for i in range(0, len(stories)) if i in selected_story_ids]


# total_qs = 0
# for story in stories:
#     total_qs += get_num_questions(story)
# print("Orig questions ", total_qs)
# total_qs = 0
# for story in final_story:
#     total_qs += get_num_questions(story)
# print("new num questions ", total_qs)

with open(folder+"/sampled/"+file_name+".txt","w") as f:
    for story in final_story:
        for line in story:
            f.write(line)

with open(folder+"/sampled/"+file_name+".trace","w") as f:
    for trace in final_trace:
        f.write(trace)




