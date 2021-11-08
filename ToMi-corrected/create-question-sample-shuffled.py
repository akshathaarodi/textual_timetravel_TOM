import random
file_data = []
file_trace = []
file_name = "test"
random.seed(1)
with open('data/'+file_name+'.txt', "r") as f:
    file_data = f.readlines()
with open('data/'+file_name+'.trace', "r") as f:
    file_trace = f.readlines()
question_lengths = []

required_questions = 500
required_unknowns = 2

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
location_stories = []
non_location_stories = []
for i in range(0, len(stories)):
    story = stories[i]
    story_done = False
    for line in story:
        if line.find("?")>-1 and line.find("location")>-1 and not story_done:
            #This is one of the stories with location questions
            location_stories.append(i)
            story_done = True;

#Location stories have ids of what is the location of X questions
print("location stories are:")
print(location_stories)

added_story_id = []
total_questions = 0

cc = 0
while total_questions < required_questions:
    cc += 1
    #print("CC :",cc)
    #print("total_questions ", total_questions)
    select_story = location_stories[random.randint(0,len(location_stories)-1)]
    #print("select_story: ",select_story)
    if select_story in added_story_id:
        #print("in added")
        continue
    added_story_id.append(select_story)
    total_questions += question_lengths[select_story]

print("added_story_id")
print(added_story_id)
# Randomly sample 1000 questions from location stories
print("total questions is")
print(total_questions)

# Get non location stories:
for i in range(0, len(stories)+1):
    if i not in location_stories:
        non_location_stories.append(i)
non_location_stories = [x for x in range(0, len(stories)) if x not in location_stories]

print("non_location_stories")
print(non_location_stories)

final_story_ids = non_location_stories + added_story_id
final_story_ids.sort()

print("Final story ids")
print(final_story_ids)

final_story = [stories[i] for i in range(0, len(stories)) if i in final_story_ids]
#final_trace = [file_trace[i] for i in range(0, len(stories)) if i in final_story_ids]


# for i in range(0, len(stories)):
#     if i in final_story_ids:
#         print("i is ", i)
#         final_story.append(stories[i])

print("orig stories ",len(stories))
print("new stories ",len(final_story))

total_qs = 0
for story in stories:
    total_qs += get_num_questions(story)
print("Orig questions ", total_qs)
total_qs = 0
for story in final_story:
    total_qs += get_num_questions(story)
print("new num questions ", total_qs)

with open("data/sampled"+file_name+".txt","w") as f:
    for story in final_story:
        for line in story:
            f.write(line)

# with open("data/sampled"+file_name+".trace","w") as f:
#     for trace in final_trace:
#         f.write(trace)