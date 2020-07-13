from mpi4py import MPI
import json
import mmap
import contextlib

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
twitter_f_path = "smallTwitter.json"
grid_f_path = "/melbGrid.json"

comm.Barrier()

# read the file 'melbGrid'
if comm_rank == 0:
    f = open(grid_f_path, 'r')
    features_list = json.loads(f.read())['features']
    f.close()
else:
    features_list = None
# broadcast the grid info
local_feature = comm.bcast(features_list, root=0)

# find margin
marginX = []
marginY = []
coordinator_x = []
coordinator_y = []
for feature in local_feature:
    x_max = feature['properties']['xmax']
    y_min = feature['properties']['ymin']
    if x_max not in coordinator_x:
        coordinator_x.append(x_max)
        marginX.append(feature['properties']['id'])
    if y_min not in coordinator_y:
        coordinator_y.append(y_min)
        marginY.append(feature['properties']['id'])

# initialization
local_twitter_Count = []
local_twitter_hashtags = []
for i in range(len(local_feature)):
    local_twitter_Count.append([local_feature[i]['properties']['id'], 0])   # twitter count
    local_twitter_hashtags.append([local_feature[i]['properties']['id'], {}])   # hashtags count


# find hashtags
def find_hashtags(text):
    hashtags_list = []
    text_item = text.split()
    for item in text_item:
        if item[0] == '#' and len(item) != 1:
            index = text.find(item)
            if index != 0 and (index + len(item) != len(text)):
                #if text[index-1] == ' ' and text[index+len(item)] == ' ':
                    if item not in hashtags_list:
                        hashtags_list.append(item)
    return hashtags_list


# process location
def process_location(local_data):
    if local_data['doc']['coordinates']:        # the location exists
        x = local_data['doc']['coordinates']['coordinates'][0]
        y = local_data['doc']['coordinates']['coordinates'][1]
        process_text(x, y, local_data)
    elif local_data['doc']['geo']:
        y = local_data['doc']['geo']['coordinates'][0]
        x = local_data['doc']['geo']['coordinates'][1]
        process_text(x, y, local_data)


# process text
def process_text(x, y, local_data):
    index = -1      # the index of the grid location
    for feature in local_feature:
        index += 1
        if (x > feature['properties']['xmin'] and x <= feature['properties']['xmax']) or \
            (x == feature['properties']['xmin'] and feature['properties']['id'] in marginY):
            if (y >= feature['properties']['ymin'] and y < feature['properties']['ymax']) or \
             (y == feature['properties']['ymax'] and feature['properties']['id'] in marginX):
                local_twitter_Count[index][1] += 1     # increase the twitter count by 1

                tag_list = find_hashtags(local_data['doc']['text'])
                if len(tag_list) > 0:  # if twitter is with hashtags
                    for tag in tag_list:
                        if (tag.lower()) in local_twitter_hashtags[index][1]:
                            local_twitter_hashtags[index][1][tag.lower()] += 1
                        else:
                            local_twitter_hashtags[index][1][tag.lower()] = 1


f = open(twitter_f_path, 'r', encoding="utf-8")
with contextlib.closing(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)) as m:
    m.readline()
    line_num = -1   # initialize the line number
    while True:
        line_num += 1
        line_byte = m.readline().strip()
        if line_num % comm_size == comm_rank:     # parallel
            line = str(line_byte, encoding='utf-8')
            if line.strip().rstrip(',') == "]}":  # ignore the last line
                break
            local_data = json.loads(line.strip().rstrip(','))  # parse Json line
            process_location(local_data)
        if m.tell() == m.size():
            break


# combine twitter count
def union_count(obj1, obj2):
    for index in range(len(obj1)):
        obj1[index][1] = obj1[index][1] + obj2[index][1]
    return obj1


# combine hashtags count
def union_hashtags(obj1, obj2):
    for index in range(len(obj1)):
        for key in obj2[index][1].keys():
            if key in obj1[index][1]:
                obj1[index][1][key] = obj1[index][1][key] + obj2[index][1][key]
            else:
                obj1[index][1][key] = obj2[index][1][key]
    return obj1


comm.Barrier()
# combine local data
twitter_count = comm.reduce(local_twitter_Count, root=0, op=union_count)
twitter_hashtags = comm.reduce(local_twitter_hashtags, root=0, op=union_hashtags)
# sort, transform and print
if comm_rank == 0:
    twitter_hashtags_final = []
    twitter_count.sort(key=lambda twitter: twitter[1], reverse=True)

    # search for the top 5 hashtags
    for i in range(len(twitter_hashtags)):
        temp_list = sorted(twitter_hashtags[i][1].items(), key=lambda tags: tags[1], reverse=True)
        if len(temp_list) > 0:
            value_temp = temp_list[0][1] + 1
            rank = 0
            index = -1
            for item in temp_list:
                value = item[1]
                if value != value_temp:
                    rank += 1
                    if rank > 5:
                        break
                    else:
                        index += 1
                        value_temp = value
                else:
                    index += 1
            final_list = temp_list[:index+1]
            twitter_hashtags_final.append([local_feature[i]['properties']['id'], tuple(final_list)])
        else:
            twitter_hashtags_final.append([local_feature[i]['properties']['id'], ()])
    
    # print Twitter count
    for i in range(len(local_feature)):
        if i != len(local_feature)-1:
            print(str(twitter_count[i][0]) + ": " + str(twitter_count[i][1]) + " posts,")
        else:
            print(str(twitter_count[i][0]) + ": " + str(twitter_count[i][1]) + " posts;")

    print("-----------------------------------------------------------------------------------")

    # print hashtags count
    for i in range(len(twitter_count)):
        for tag in twitter_hashtags_final:
            if twitter_count[i][0] == tag[0]:
                tagCountStr = str(tag[1]).replace('\'', '')
                if i != len(local_feature)-1:
                    print(str(tag[0]) + ": " + tagCountStr)
                else:
                    print(str(tag[0]) + ": " + tagCountStr + ";")
