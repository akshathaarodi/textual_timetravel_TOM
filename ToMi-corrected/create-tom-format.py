import random
file_data = []
file_trace = []
file_name = "test"

with open('data/'+file_name+'.txt', "r") as f:
    file_data = f.readlines()
with open('data/'+file_name+'.trace', "r") as f:
    file_trace = f.readlines()
question_lengths = []


first_line = file_data[0]
questions = 0
stories = []
story = []
story_ind = 0 

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
question_lengths.append(questions)
question_lengths = question_lengths[1:]
print("stories are")
print(stories)
print("story_lengths are")
print(question_lengths)


memory_stories = []
reality_stories = []
tb_fo_stories = []
tb_so_stories = []
fb_fo_stories = []
fb_so_stories = []


final_story = []
location_stories = []
non_location_stories = []
for i in range(0, len(stories)):
    story = stories[i]
    trace = file_trace[i]
    trace_word = trace.split(',')[-2]
    if trace_word.find("memory") > -1:
        memory_stories.append(story)
    if trace_word.find("reality") > -1:
        reality_stories.append(story)
    if trace_word.find("no_tom") > -1:
        if trace_word.find("first_order") > -1:
            tb_fo_stories.append(story)
        elif trace_word.find("second_order") > -1:
            tb_so_stories.append(story)
    elif trace_word.find("tom") > -1:
        if trace_word.find("first_order") > -1:
            fb_fo_stories.append(story)
        elif trace_word.find("second_order") > -1:
            fb_so_stories.append(story)


with open("data/memory_"+file_name+"_test.txt","w") as f:
    for story in memory_stories:
        for line in story:
            f.write(line)

with open("data/reality_"+file_name+"_test.txt","w") as f:
    for story in reality_stories:
        for line in story:
            f.write(line)

with open("data/tb_fo_"+file_name+"_test.txt","w") as f:
    for story in tb_fo_stories:
        for line in story:
            f.write(line)

with open("data/tb_so_"+file_name+"_test.txt","w") as f:
    for story in tb_so_stories:
        for line in story:
            f.write(line)


with open("data/fb_fo_"+file_name+"_test.txt","w") as f:
    for story in fb_fo_stories:
        for line in story:
            f.write(line)

with open("data/fb_so_"+file_name+"_test.txt","w") as f:
    for story in fb_so_stories:
        for line in story:
            f.write(line)