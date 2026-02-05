import os

import s3fs
from dotenv import load_dotenv

import pandas as pd
from openai import OpenAI
import numpy.random as npr

load_dotenv(override=True)

# Import notice

fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.environ["AWS_ACCESS_KEY_ID"], 
    secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
    token = os.environ["AWS_SESSION_TOKEN"])

NOTICES_PATH = "projet-ape/notices/Notices-NAF2025-FR.parquet"

COLUMNS_TO_KEEP = [
    "ID",
    "CODE",
    "NAME",
    "FINAL",
    "Implementation_rule",
    "Includes",
    "IncludesAlso",
    "Excludes",
    "text_content",
]

df_full_notice = pd.read_parquet(NOTICES_PATH, filesystem=fs)[COLUMNS_TO_KEEP]

# Extract relevant information
df_notice_final_level = df_full_notice[df_full_notice["FINAL"] == 1].drop(columns=["FINAL"])

#First, we just keep one value
notice_example = df_notice_final_level.sample(1).to_dict(orient='list')
for key, value in notice_example.items():
    notice_example[key] = value[0]

# Set up LLM

LLM_LAB_API_KEY = os.environ["LLM_LAB_API_KEY"]
client = OpenAI(api_key=LLM_LAB_API_KEY, base_url="https://llm.lab.sspcloud.fr/api")

def ask_model(system_prompt, user_prompt, temperature):
    response = client.chat.completions.create(
        model="gpt-oss:20b",  # Ou gpt-4, etc.
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt}
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content



### Ask LLM
SYSTEM_PROMPT = "You are the representative of a company."


# Create prompt
user_prompt = "Write in a short sentence in English the official economic activity of your company to fill out the wording of a form."

# Add title
if notice_example["NAME"]:
    user_prompt += "\n The main activity of the company is about " + notice_example["NAME"]

# Add specifications
if notice_example["Includes"]:
    all_includes = notice_example["Includes"].split("\n")[1:]
    nb_includes = len(all_includes)
    random_includes = npr.choice(all_includes, npr.randint(1,nb_includes, ), replace= False)
    user_prompt += "\n Your comapny is specialized in these fields:"
    for include in random_includes:
        user_prompt += "\n" + include

if notice_example["IncludesAlso"]:
    all_includes_also = notice_example["IncludesAlso"].split("\n")[1:]
    nb_includes_also = len(all_includes_also)
    random_includes_also = npr.choice(all_includes_also, npr.randint(0,nb_includes_also), replace= False)
    user_prompt += "\n Your comapny is specialized in these fields:"
    for include_also in random_includes_also:
        user_prompt += "\n" + include_also

if notice_example["Excludes"]:
    all_excludes = notice_example["Excludes"].split("\n")[1:]
    nb_excludes = len(all_excludes)
    random_excludes = npr.choice(all_excludes, npr.randint(1,nb_excludes, ), replace= False)
    user_prompt += "\n Your comapny is specialized in these fields:"
    for exclude in random_excludes:
        user_prompt += "\n" + exclude

user_prompt += "\n The wording must be impersonal and official."

TEMPERATURE = 0.7

print("NACE Code:", notice_example["CODE"], notice_example["NAME"])
print("Prompt: \n", user_prompt)
print("Temperature: ", TEMPERATURE)
response = ask_model(SYSTEM_PROMPT, user_prompt, TEMPERATURE)
print(f"Model : {response}")








