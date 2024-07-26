import git
import re
import time
import pandas as pd
import os
import glob
from nltk.stem import WordNetLemmatizer
import nltk

def normalize_word(sentence):
    lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    res = []
    for word, pos in pos_tags:
        if pos.startswith('J'):
            tmp = lemmatizer.lemmatize(word.lower(), 'a')
        elif pos.startswith('V'):
            tmp = lemmatizer.lemmatize(word.lower(), 'v')
        elif pos.startswith('N'):
            tmp = lemmatizer.lemmatize(word.lower(), 'n')
        elif pos.startswith('R'):
            tmp = lemmatizer.lemmatize(word.lower(), 'r')
        else:
            tmp = lemmatizer.lemmatize(word.lower())
        res.append(tmp)
    return ' '.join(res)

def remove_java_comments(code):
    code = re.sub(r'/\*(.|\n)*?\*/', '', code)
    code = re.sub(r'/\*\*.*?\*/', '', code)
    return code

def isToDelete(text):
    if text.startswith('+ /*') or text.startswith('+ *') or text.startswith('+ */') or text.startswith('+ //') or text.startswith('+ /**') or text.startswith('- /*') or text.startswith('- *') or text.startswith('- */') or text.startswith('- //') or text.startswith('- /**'):
        return True
    else:
        return False

def isOnlyAddOrDeleteSymbol(text):
    char_list = [char for char in text]
    if len(char_list) == 1:
        return True

    other_list = char_list[1:]
    other_str = ''.join(other_list)
    return other_str.isspace()


def getNeededLines(code_string):
    temp_list = code_string.split('\n')
    add_or_delete_lines = []
    for item in temp_list:
        if item.startswith('+') or item.startswith('-'):
            add_or_delete_lines.append(item)

    fine_list = []
    for item in add_or_delete_lines:
        if not isOnlyAddOrDeleteSymbol(item):
            fine_list.append(item)

    code_string = '\n'.join(fine_list)

    cleaned_code = remove_java_comments(code_string)
    cleaned_code = cleaned_code.strip()

    modified_string = re.sub(r'^\+\s*\n', '', cleaned_code, flags=re.MULTILINE)
    modified_string = re.sub(r'^\-\s*\n', '', cleaned_code, flags=re.MULTILINE)
    modified_string = re.sub(r'^\+(?=[^\s])', '+ ', modified_string, flags=re.MULTILINE)
    modified_string = re.sub(r'^\+(\s+)(?=\S)', '+ ', modified_string, flags=re.MULTILINE)
    modified_string = re.sub(r'^\+\s*$', '', modified_string, flags=re.MULTILINE)
    modified_string = re.sub(r'^\-(?=[^\s])', '- ', modified_string, flags=re.MULTILINE)
    modified_string = re.sub(r'^\-(\s+)(?=\S)', '- ', modified_string, flags=re.MULTILINE)
    modified_string = re.sub(r'^\-\s*$', '', modified_string, flags=re.MULTILINE)

    check_list = modified_string.split('\n')
    after_delete_list = []
    for item in check_list:
        if isToDelete(item) == False:
            after_delete_list.append(item)

    result_list = []
    for item in after_delete_list:
        if item.startswith('+'):
            temp_list = item.split('+ ')
            temp_str = ''.join(temp_list)
            if not temp_str.startswith('+'):
                final_str = '+ ' + temp_str
                result_list.append(final_str)
        elif item.startswith('-'):
            temp_list = item.split('- ')
            temp_str = ''.join(temp_list)
            if not temp_str.startswith('-'):
                final_str = '- ' + temp_str
                result_list.append(final_str)

    final_list = []
    for item in result_list:
        if not item.startswith('+ //') or item.startswith('- //'):
            final_list.append(item)

    final_str = '\n'.join(final_list)

    return final_str

def getDiffType(diff_text):
    lines = diff_text.split('\n')
    mode = lines[1]

    if mode.startswith('new file mode'):
        return 0
    elif mode.startswith('deleted file mode'):
        return 1
    else:
        return 2


def getDiffSample(diff):
    diff_lines = diff.split('\n')
    index_list = []
    for idx, line in enumerate(diff_lines):
        if line.startswith('diff --git'):
            index_list.append(idx)

    diff_part_list = []
    for i in range(len(index_list)):
        if i != len(index_list) - 1:
            current_idx = index_list[i]
            next_idx = index_list[i + 1]
            diff_part = diff_lines[current_idx:next_idx]
            diff_part_str = '\n'.join(diff_part)
            diff_part_list.append(diff_part_str)
        else:
            diff_part = diff_lines[index_list[-1]:]
            diff_part_str = '\n'.join(diff_part)
            diff_part_list.append(diff_part_str)

    diff_java_list = []
    for diff_content in diff_part_list:
        temp_list = diff_content.split('\n')
        if '.java' in temp_list[0]:
            diff_java_list.append(diff_content)

    diff_sample_list = []

    if not len(diff_java_list) == 0:
        for item in diff_java_list:
            if getDiffType(item) == 0 or getDiffType(item) == 1:
                diff_one_list = item.split('\n')
                head_info_list = diff_one_list[:5]
                diff_body_list = diff_one_list[5:]
                diff_body_str = '\n'.join(diff_body_list)

                changed_file_info = head_info_list[0]
                target_changed_file_name = changed_file_info.split(' ')[2].split('/')[-1]

                target_changed_file_name = target_changed_file_name.replace('.', ' . ')

                if getDiffType(item) == 0:
                    head_text = 'new file <nl> ppp ' + target_changed_file_name + ' <nl> '
                    neededLinesStr = getNeededLines(diff_body_str)
                    temp_lines = neededLinesStr.split('\n')
                    tem_list = []
                    for item in temp_lines:
                        item = item.replace('.', ' . ').replace(';', ' ;')
                        item = item + ' <nl>'
                        tem_list.append(item)

                    change_body_str = ' '.join(tem_list)
                    one_diff_sample = head_text + change_body_str
                    # print(one_diff_sample)
                    diff_sample_list.append(one_diff_sample)
                elif getDiffType(item) == 1:
                    head_text = 'deleted file <nl> mmm ' + target_changed_file_name + ' <nl> '
                    neededLinesStr = getNeededLines(diff_body_str)
                    temp_lines = neededLinesStr.split('\n')
                    tem_list = []
                    for item in temp_lines:
                        item = item.replace('.', ' . ').replace(';', ' ;')
                        item = item + ' <nl>'
                        tem_list.append(item)

                    change_body_str = ' '.join(tem_list)
                    one_diff_sample = head_text + change_body_str
                    # print(one_diff_sample)
                    diff_sample_list.append(one_diff_sample)

            elif getDiffType(item) == 2:
                diff_one_list = item.split('\n')
                head_info_list = diff_one_list[:4]
                diff_body_list = diff_one_list[4:]
                diff_body_str = '\n'.join(diff_body_list)

                changed_file_info = head_info_list[0]
                target_changed_file_name = changed_file_info.split(' ')[2].split('/')[-1]
                # print(target_changed_file_name)
                target_changed_file_name = target_changed_file_name.replace('.', ' . ')

                head_text = 'mmm ' + target_changed_file_name + ' <nl> ppp ' + target_changed_file_name + ' <nl>'
                neededLinesStr = getNeededLines(diff_body_str)
                temp_lines = neededLinesStr.split('\n')
                tem_list = []
                for item in temp_lines:
                    item = item.replace('.', ' . ').replace(';', ' ;')
                    item = item + ' <nl>'
                    tem_list.append(item)

                change_body_str = ' '.join(tem_list)
                one_diff_sample = head_text + ' ' + change_body_str
                # print(one_diff_sample)
                diff_sample_list.append(one_diff_sample)

    return diff_sample_list


def isTodeleteMsg(message):
    message = str(message)
    tokens = message.split(' ')
    first_token = tokens[0]

    if '/' in first_token or '@' in first_token:
        return True
    elif 'typo' in message or 'doc' in message or '.txt' in message or '.md' in message or '.sh' in message:
        return True
    else:
        return False


def normalizeString(string):
    normalized_string = string.encode('utf-8' , 'replace').decode('utf-8')
    return normalized_string

def normalizeMsg1(msg):
    msg = str(msg)
    tokens = msg.split(' ')
    res_list = []
    for token in tokens:
        if not (token.startswith('[') or token.startswith('(')):
            res_list.append(token)

    return ' '.join(res_list)

def normalizeMsg2(msg):

    msg = str(msg)
    tokens = msg.split(' ')
    first_token = tokens[0]

    if ':' in first_token and first_token != 'fix:':
        res_list = tokens[1:]
    else:
        res_list = tokens

    return ' '.join(res_list)

def normalizeMsg3(msg):

    msg = str(msg)

    if '*' in msg:
        msg.replace('*' , '')
        msg = msg.strip()
        return msg
    elif '{' in msg:
        msg.replace('{' , '')
        msg = msg.strip()
        return msg
    elif '}' in msg:
        msg.replace('}', '')
        msg = msg.strip()
        return msg
    elif '+' in msg:
        msg.replace('+', '')
        msg = msg.strip()
        return msg
    else:
        return msg


if __name__ == '__main__':

    s_time = time.time()

    Max_diff_len = 100
    Min_diff_len = 30
    Max_msg_len = 30
    Min_msg_len = 3

    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    # nltk.download('omw-1.4')

    basePath = 'your path to read the projects'
    project_names = [name for name in os.listdir(basePath) if os.path.isdir(os.path.join(basePath, name))]

    project_num = len(project_names)

    projects_data_num_origin = 0
    projects_data_num_filtered = 0

    for project_name in project_names:
        repo_path = os.path.join(basePath , project_name)

        print(f'===== Collecting Project: {project_name} =====')


        repo = git.Repo(repo_path)

        remote_url = repo.remotes.origin.url
        owner_name = remote_url.split('/')[-2]
        repo_name = remote_url.split('/')[-1].split('.')[0]
        full_name = owner_name + '/' + repo_name


        commits = list(repo.iter_commits())
        commits = commits[::-1]

        commit_diff_list = []
        commit_msg_list = []
        commit_sha_list = []
        commit_timestamp_list = []
        commit_author_list = []

        for idx, commit in enumerate(commits):

            commit_message = commit.message.strip()
            temp_list = commit_message.split('\n')
            commit_message = temp_list[0]
            # commit_message = normalizeMsg1(commit_message)
            commit_message = normalize_word(str(commit_message))
            # commit_message = normalizeMsg2(str(commit_message))
            # commit_message = normalizeMsg3(str(commit_message))

            commit_id = commit.hexsha
            parent_commit_id = commit.parents[0].hexsha if commit.parents else None

            commit_time = commit.committed_datetime

            commit_author = commit.author.name

            if parent_commit_id:
                diff = repo.git.diff(parent_commit_id, commit_id)
                res_list = getDiffSample(diff)

                final_diff_info = ''
                if len(res_list) != 0:
                    diff_str = ' '.join(res_list)
                    diff_token_list = diff_str.split(' ')

                    if len(diff_token_list) > Max_diff_len:
                        diff_token_list = diff_token_list[:Max_diff_len-1]
                        diff_token_list.append('<nl>')

                    final_diff_info = ' '.join(diff_token_list)

                commit_diff_list.append(final_diff_info)
                commit_msg_list.append(commit_message)
                commit_sha_list.append(commit_id)
                commit_timestamp_list.append(commit_time)
                commit_author_list.append(commit_author)

        if len(commit_diff_list) == len(commit_msg_list) == len(commit_sha_list) == len(commit_timestamp_list) == len(commit_author_list):
            print(f'The initial number:{len(commit_diff_list)}')
            projects_data_num_origin = projects_data_num_origin + len(commit_diff_list)

        # Filter 1: diff is not empty
        not_none_diff_index = []
        for idx , item in enumerate(commit_diff_list):
            if not item == '':
                not_none_diff_index.append(idx)

        commit_diff_list_filtered1 = []
        commit_msg_list_filtered1 = []
        commit_sha_list_filtered1 = []
        commit_timestamp_list_filtered1 = []
        commit_author_list_filtered1 = []

        for index , diff in enumerate(commit_diff_list):
            if index in not_none_diff_index:
                commit_diff_list_filtered1.append(diff)

        for index , msg in enumerate(commit_msg_list):
            if index in not_none_diff_index:
                commit_msg_list_filtered1.append(msg)

        for index , sha in enumerate(commit_sha_list):
            if index in not_none_diff_index:
                commit_sha_list_filtered1.append(sha)

        for index , timestamp in enumerate(commit_timestamp_list):
            if index in not_none_diff_index:
                commit_timestamp_list_filtered1.append(timestamp)

        for index , author in enumerate(commit_author_list):
            if index in not_none_diff_index:
                commit_author_list_filtered1.append(author)

        if len(commit_diff_list_filtered1) == len(commit_msg_list_filtered1) == len(commit_sha_list_filtered1) == len(commit_timestamp_list_filtered1) == len(commit_author_list_filtered1):
            print(f'size after filter 1:{len(commit_diff_list_filtered1)}')


        # filter2: remove merge and revert commits
        not_rollback_and_merge_index = []
        for i in range(len(commit_msg_list_filtered1)):
            msg = commit_msg_list_filtered1[i]
            if ('merge' not in msg) and ('Merge' not in msg) and ('revert' not in msg) and ('Revert' not in msg):
                not_rollback_and_merge_index.append(i)

        commit_diff_list_filtered2 = []
        commit_msg_list_filtered2 = []
        commit_sha_list_filtered2 = []
        commit_timestamp_list_filtered2 = []
        commit_author_list_filtered2 = []

        for index, diff in enumerate(commit_diff_list_filtered1):
            if index in not_rollback_and_merge_index:
                commit_diff_list_filtered2.append(diff)

        for index, msg in enumerate(commit_msg_list_filtered1):
            if index in not_rollback_and_merge_index:
                commit_msg_list_filtered2.append(msg)

        for index, sha in enumerate(commit_sha_list_filtered1):
            if index in not_rollback_and_merge_index:
                commit_sha_list_filtered2.append(sha)

        for index, timestamp in enumerate(commit_timestamp_list_filtered1):
            if index in not_rollback_and_merge_index:
                commit_timestamp_list_filtered2.append(timestamp)

        for index, author in enumerate(commit_author_list_filtered1):
            if index in not_rollback_and_merge_index:
                commit_author_list_filtered2.append(author)

        if len(commit_diff_list_filtered2) == len(commit_msg_list_filtered2) == len(commit_sha_list_filtered2) == len(commit_timestamp_list_filtered2) == len(commit_author_list_filtered2):
            print(f'size after filter2:{len(commit_diff_list_filtered2)}')


        # filter3: remove commit message with specific patterns
        commit_save_index = []
        for i in range(len(commit_msg_list_filtered2)):
            msg = commit_msg_list_filtered2[i]
            if not isTodeleteMsg(msg):
                commit_save_index.append(i)

        commit_diff_list_filtered3 = []
        commit_msg_list_filtered3 = []
        commit_sha_list_filtered3 = []
        commit_timestamp_list_filtered3 = []
        commit_author_list_filtered3 = []

        for index, diff in enumerate(commit_diff_list_filtered2):
            if index in commit_save_index:
                commit_diff_list_filtered3.append(diff)

        for index, msg in enumerate(commit_msg_list_filtered2):
            if index in commit_save_index:
                commit_msg_list_filtered3.append(msg)

        for index, sha in enumerate(commit_sha_list_filtered2):
            if index in commit_save_index:
                commit_sha_list_filtered3.append(sha)

        for index, timestamp in enumerate(commit_timestamp_list_filtered2):
            if index in commit_save_index:
                commit_timestamp_list_filtered3.append(timestamp)

        for index, author in enumerate(commit_author_list_filtered2):
            if index in commit_save_index:
                commit_author_list_filtered3.append(author)

        if len(commit_diff_list_filtered3) == len(commit_msg_list_filtered3) == len(commit_sha_list_filtered3) == len(commit_timestamp_list_filtered3) == len(commit_author_list_filtered3):
            print(f'size of filter3:{len(commit_diff_list_filtered3)}')


        # filter4: remove commit message and diff with specific length
        len_satisfy_index = []

        for i in range(len(commit_msg_list_filtered3)):
            msg = commit_msg_list_filtered3[i]
            diff_text = commit_diff_list_filtered3[i]
            msg_tokens = msg.split(' ')
            diff_text_tokens = diff_text.split(' ')

            if (Min_msg_len <= len(msg_tokens) <= Max_msg_len) and (Min_diff_len <= len(diff_text_tokens) <= Max_diff_len):
                len_satisfy_index.append(i)

        commit_diff_list_filtered4 = []
        commit_msg_list_filtered4 = []
        commit_sha_list_filtered4 = []
        commit_timestamp_list_filtered4 = []
        commit_author_list_filtered4 = []

        for index, diff in enumerate(commit_diff_list_filtered3):
            if index in len_satisfy_index:
                commit_diff_list_filtered4.append(diff)

        for index, msg in enumerate(commit_msg_list_filtered3):
            if index in len_satisfy_index:
                commit_msg_list_filtered4.append(msg)

        for index, sha in enumerate(commit_sha_list_filtered3):
            if index in len_satisfy_index:
                commit_sha_list_filtered4.append(sha)

        for index, timestamp in enumerate(commit_timestamp_list_filtered3):
            if index in len_satisfy_index:
                commit_timestamp_list_filtered4.append(timestamp)

        for index, author in enumerate(commit_author_list_filtered3):
            if index in len_satisfy_index:
                commit_author_list_filtered4.append(author)

        if len(commit_diff_list_filtered4) == len(commit_msg_list_filtered4) == len(commit_sha_list_filtered4) == len(commit_timestamp_list_filtered4) == len(commit_author_list_filtered4):
            print(f'size after filter4:{len(commit_diff_list_filtered4)}')
            projects_data_num_filtered = projects_data_num_filtered + len(commit_diff_list_filtered4)



        for index , diff in enumerate(commit_diff_list_filtered4):
            normalized_diff = normalizeString(diff)
            commit_diff_list_filtered4[index] = normalized_diff

        for index , msg in enumerate(commit_msg_list_filtered4):
            normalized_msg = normalizeString(msg)
            commit_msg_list_filtered4[index] = normalized_msg


        # save the filtered data
        data = {'commitId':commit_sha_list_filtered4 , 'commitTime':commit_timestamp_list_filtered4 ,'author':commit_author_list_filtered4, 'message':commit_msg_list_filtered4 , 'diff':commit_diff_list_filtered4}
        df = pd.DataFrame(data)
        save_path = os.path.join('your path to save data after filtering' , f'filtered_data_{project_name}.csv')
        df.to_csv(save_path , encoding='utf-8' , index=False)

        print('*' * 200)


    e_time = time.time()
    print(f'Total: {project_num} projects')
    print(f'initial size:{projects_data_num_origin}')
    print(f'size after filter:{projects_data_num_filtered}')
    print(f'Total time cost:{e_time - s_time} s')

