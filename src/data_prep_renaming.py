from os import walk
from os import listdir
import os

print("Renaming starts")

my_path = "examples/extra/MIR"

def clean_up_desc(desc_with_suffix):
    desc = desc_with_suffix.split(".", 1)[0]
    desc = desc.replace("-", "")
    desc = desc.replace("_", "")
    return desc

def create_new_file_name(cotton_content, desc):
    polyester_content = 100 - int(cotton_content)
    # extra marker, so that there are no overlaps 
    date = "240613"
    new_name = "s_" + str(cotton_content) + "_" + str(polyester_content) + "_" + desc + "_" + date + ".txt"
    return new_name

def starts_with_prefix(file):
    return file.startswith("nir") or file.startswith("mir")

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
                cotton_content = splitted_name[0]
                # We just look for the description, we ignore polyester content
                desc_with_suffix = splitted_name[1].split("_", 1)[1]
                desc = clean_up_desc(desc_with_suffix)
                if "_" in cotton_content:
                    cotton_content = cotton_content.split("_")[0]
                    cotton_content = int(cotton_content) + 1
                new_name = create_new_file_name(cotton_content, desc)
            else:
                cotton_content = 50
                desc = clean_up_desc(file)
                new_name = create_new_file_name(cotton_content, desc)
            old_file = os.path.join(my_path, file)
            new_file = os.path.join(my_path, new_name)
            print(f"{file} will be renamed to {new_name}")
            #os.rename(old_file, new_file)
