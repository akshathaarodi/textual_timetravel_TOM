import json
import os
import random as r
import string

#test_files = glob.glob("data/tom/world_large_nex_1000_0/*test_test.txt")
#val_files = glob.glob("data/tom/world_large_nex_1000_0/*val_test.txt")
#train_files = glob.glob("data/tom/world_large_nex_1000_0/*train.txt")
#train_files = ['data/tom/world_large_nex_1000_0/tom

files = ['train', 'val', 'test']
out_dir = "data/data-after-fix"
for file in files:
    file_data = []
    #file_name = 'fb_all_val.txt'
    file_name = os.path.join(out_dir, f"{file}.txt")
    print('file is', file_name)

    with open(file_name, 'r') as f:
        file_data += f.readlines()
    #print("filedata is", file_data)
    context = ""
    counter = 0
    data_dicts = []
    first_encountered = 0
    paragraph = []
    qas = []
    semi_para = []
    for sent in file_data:
        sent_split = sent.split('\t')
        cleaned_sent = sent_split[0]
        first_num = cleaned_sent.split(' ')[0]
        cleaned_sent = cleaned_sent.lstrip(string.digits).lstrip().rstrip()
        semi_para.append(cleaned_sent)

        if cleaned_sent[-1] == '?':
            id = ''.join(r.choices(string.ascii_uppercase + string.digits, k=24))
            answer = sent_split[1]
            support = int(sent_split[2])
            # print("context is")
            # print(context)
            answer_start = context.index(answer)
            # ans_line = semi_para[support-1]
            # print('file is',files[i])
            # print("support is",support)
            # print("semi para is", semi_para)
            # print("answer line is", ans_line)
            # print("answer is", answer)
            # answer_start = context.index(ans_line)
            # answer_start += ans_line.index(answer)
            semi_para = []
            question = cleaned_sent
            answers = [{"answer_start": answer_start, "text": answer}]
            if context.count(answer) > 1:
                print('answer is found more than once')
                print('question is', question)
                print('answers is', answers)
            #         print('verify answer start', context[answer_start:answer_start+len(answer)])
            # if answer !=  context[answer_start:answer_start+len(answer)]:
            #     print('mismatch found')
            #     print('question is', question)
            #     print('answers is', answers)
            #     print('verify answer start', context[answer_start:answer_start+len(answer)])
            qas += [{"answers": answers, "question": question, "id": id}]
            title = "FB_test" + str(counter)
            counter += 1
            paragraph = [{"context": context, "qas": qas}]
            data_dict = {"title": title, "paragraphs": paragraph}
            context = ''
            # print('datadict is', data_dict)
            data_dicts.append(data_dict)
            paragraph = []
            qas = []
        else:
            if cleaned_sent[-1] != '.':
                cleaned_sent += '.'
            context += cleaned_sent + " "
    #         print('context is')
    #         print(context)

    outjson = os.path.join(out_dir, f"{file}.json")
    j = {"data": data_dicts}
    with open(outjson, "w") as outfile:
        json.dump(j, outfile)