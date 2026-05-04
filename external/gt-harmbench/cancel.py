from openai import OpenAI
client = OpenAI()

batch_id = "batch_6956ae6e7c488190be92f1b687270caa"
result = client.batches.cancel("batch_6956ae6e7c488190be92f1b687270caa")

# check status
# result = client.batches.retrieve(batch_id)
print(result)