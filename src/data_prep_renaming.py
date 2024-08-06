from os import walk
from os import listdir
import os

print("Renaming starts")

my_path = "../input/clean_txt/MIR/240806_mir"

def clean_up_desc(desc_with_suffix):
    desc = desc_with_suffix.split(".", 1)[0]
    desc = desc.replace("-", "")
    print(desc)
    desc = desc.replace("sample", "specimen")
    desc = desc.replace("q", "area")
    desc = desc.replace("petcotton_", "")
    desc = desc.replace("_0", "")
    print(desc)
    #desc = desc.replace("_", "")
    return desc

def create_new_file_name(polyester_content, desc):
    cotton_content = 100 - int(polyester_content)
    # extra marker, so that there are no overlaps 
    date = "240806"
    new_name = "s_" + str(polyester_content) + "_" + str(cotton_content) + "_" + desc + "_" + date + ".txt"
    return new_name

def starts_with_prefix(file):
    return file.startswith("vartest") or file.startswith("mir")

# r=root, d=directories, f = files
for r, d, f in os.walk(my_path):
    for file in f:
        if file.endswith(".TXT"):
            if file[0].isdigit() or starts_with_prefix(file):
                if starts_with_prefix(file):
                    file_name = file.split("_", 1)[1]
                else:
                    file_name = file
                splitted_name = file_name.split("-", 1)
                polyester_content = splitted_name[0]
                # We just look for the description, we ignore cotton content
                desc_with_suffix = splitted_name[1].split("_", 1)[1]
                desc = clean_up_desc(desc_with_suffix)
                # We round up when there is a decimal
                if "_" in polyester_content:
                    polyester_content = polyester_content.split("_")[0]
                    polyester_content = int(polyester_content) + 1
                new_name = create_new_file_name(polyester_content, desc)
            else:
                polyester_content = 50
                desc = clean_up_desc(file)
                new_name = create_new_file_name(polyester_content, desc)
            old_file = os.path.join(my_path, file)
            new_file = os.path.join(my_path, new_name)
            print(f"{file} will be renamed to {new_name}")
            os.rename(old_file, new_file)
